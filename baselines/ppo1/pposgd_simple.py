import math
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque


class EpisodeRunner:
    def __init__(self, env):
        self.env = env
        self.prevacs = []
        self.acs = []
        self.vpreds = []
        self.obs = [self.env.reset()]
        self.rews = []
        self.news = []
        self.cur_ep_ret = 0.0
        self.cur_ep_len = 0

    @property
    def ob(self):
        return self.obs[-1]

    @property
    def prevac(self):
        return self.acs[-1] if self.acs else self.env.action_space.sample()

    @property
    def terminated(self):
        return self.news[-1] if self.news else False

    def step(self, ac, vpred):
        if self.terminated:
            return True

        ob, rew, new, _ = self.env.step(ac)
        self.prevacs.append(self.prevac)  # before acs
        self.acs.append(ac)
        self.vpreds.append(vpred)
        if not new:
            self.obs.append(ob)
        self.rews.append(rew)
        self.news.append(new)
        self.cur_ep_ret += rew
        self.cur_ep_len += 1
        return new


def _create_episode_runners(env, ep_lens, timesteps):
    episode_runners = [EpisodeRunner(env)]
    env_unwrapped = env.unwrapped
    if hasattr(env_unwrapped, 'create_new_instance'):
        if not ep_lens:
            n_episodes = 10
        else:
            ep_mean_length = sum(ep_lens) / len(ep_lens)
            n_episodes = int(math.ceil(timesteps / ep_mean_length))
        episode_runners += [EpisodeRunner(env_unwrapped.create_new_instance()) for _ in range(n_episodes - 1)]
    return episode_runners


def traj_segment_generator(pi, env, horizon, stochastic):
    ep_lens = []
    while True:
        episode_runners = _create_episode_runners(env, ep_lens, horizon)

        obs = np.array([episode_runners[0].ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        ac = env.action_space.sample()
        acs = np.array([ac for _ in range(horizon)])
        prevacs = np.zeros_like(acs)
        ep_lens = []
        ep_rets = []
        i = 0
        vpred = 0.0

        while i < horizon:
            if hasattr(pi, 'batch_act'):
                ep_acs, ep_vpreds = pi.batch_act(stochastic, [ep_runner.ob for ep_runner in episode_runners])
            else:
                ep_acs, ep_vpreds = zip(*(pi.act(stochastic, ep_runner.ob) for ep_runner in episode_runners))
            ep_news = [ep_runner.step(ac, vpred) for ep_runner, ac, vpred in zip(episode_runners, ep_acs, ep_vpreds)]

            # Only take the episodes in the order they are in the runners list
            # to avoid a bias in always only taking the shortest ones
            first_not_terminated = None
            for ep_idx, (ep_runner, ep_new) in enumerate(zip(episode_runners, ep_news)):
                if not ep_new:
                    first_not_terminated = ep_idx
                    break
                max_idx = min(i + len(ep_runner.obs), horizon)
                ep_max_idx = max_idx - i
                obs[i:max_idx] = ep_runner.obs[:ep_max_idx]
                rews[i:max_idx] = ep_runner.rews[:ep_max_idx]
                vpreds[i:max_idx] = ep_runner.vpreds[:ep_max_idx]
                news[i:max_idx] = ep_runner.news[:ep_max_idx]
                acs[i:max_idx] = ep_runner.acs[:ep_max_idx]
                prevacs[i:max_idx] = ep_runner.prevacs[:ep_max_idx]
                ep_rets.append(ep_runner.cur_ep_ret)
                ep_lens.append(ep_runner.cur_ep_len)
                i = max_idx
                vpred = ep_runner.vpreds[ep_max_idx] if ep_max_idx < len(ep_runner.vpreds) else 0.0
                if i == horizon:
                    break

            if first_not_terminated is None:
                episode_runners = _create_episode_runners(env.unwrapped, ep_lens, horizon - i)
            else:
                episode_runners = episode_runners[first_not_terminated:]

        # Be careful!!! if you change the downstream algorithm to aggregate
        # several of these batches, then be sure to do a deepcopy
        yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
               "ac": acs, "prevac": prevacs, "nextvpred": vpred,
               "ep_rets": ep_rets, "ep_lens": ep_lens}

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

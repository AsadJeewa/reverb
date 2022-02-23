import reverb
import tensorflow as tf
import time
from tqdm import tqdm
import enum
import numpy as np
import argparse


class Mode(enum.Enum):
    Legacy = 1
    Trajectory = 2
    Sequence = 3


parser = argparse.ArgumentParser(description="Benchmark Writers")
parser.add_argument("--mode", action="store", type=str, required=True, help="run mode")
args = parser.parse_args()
print(args.mode)

if "leg" in str.lower(args.mode):
    mode = Mode.Legacy
elif "traj" in str.lower(args.mode):
    mode = Mode.Trajectory
elif "seq" in str.lower(args.mode):
    mode = Mode.Sequence

print(mode)
misc = "ep100_step1000_seq200_period1_"
if mode == mode.Legacy:
    alg = "legacy"
elif mode == mode.Trajectory:
    alg = "trajectory"
elif mode == mode.Sequence:
    alg = "sequence"
# alg = "legacy" if mode == mode.Legacy elif "trajectory"
env = ""
adjust = misc + alg + env + ".txt"
print(adjust)
f = open(adjust, "w")

num_episodes = 100
num_episode_steps = 1000
seq_len = 200
period_len = 1
# overlap = 2

OBSERVATION_SPEC = tf.TensorSpec([10, 10], tf.uint8)
ACTION_SPEC = tf.TensorSpec([2], tf.float32)


def agent_step(unused_timestep) -> tf.Tensor:
    return tf.cast(tf.random.uniform(ACTION_SPEC.shape) > 0.5, ACTION_SPEC.dtype)


def environment_step(unused_action) -> tf.Tensor:
    return tf.cast(
        tf.random.uniform(OBSERVATION_SPEC.shape, maxval=256), OBSERVATION_SPEC.dtype
    )


# Initialize the reverb server.
if mode == mode.Legacy:
    legacy_server = reverb.Server(
        tables=[
            reverb.Table(
                name="legacy_table",
                sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
                remover=reverb.selectors.Fifo(),
                max_size=int(1e6),
                # Sets Rate Limiter to a low number for the examples.
                # Read the Rate Limiters section for usage info.
                rate_limiter=reverb.rate_limiters.MinSize(2),
                # The signature is optional but it is good practice to set it as it
                # enables data validation and easier dataset construction. Note that
                # we prefix all shapes with a 3 as the trajectories we'll be writing
                # consist of 3 timesteps.
                signature={
                    "actions": tf.TensorSpec(
                        [*ACTION_SPEC.shape], ACTION_SPEC.dtype
                    ),  # DO NOT PUT SEUQNCE LENGTH IN SIGNATURE IF LEGACY WRITER
                    "observations": tf.TensorSpec(
                        [*OBSERVATION_SPEC.shape], OBSERVATION_SPEC.dtype
                    ),
                },
            )
        ],
        # Sets the port to None to make the server pick one automatically.
        # This can be omitted as it's the default.
        port=None,
    )

    # Initializes the reverb client on the same port as the server.
    client = reverb.Client(f"localhost:{legacy_server.port}")

    # Dynamically adds trajectories of length 3 to 'my_table' using a client writer.
    start_exe = time.time()
    with client.writer(
        seq_len
    ) as writer:  # trajectory_writer(num_keep_alive_refs=3) as writer:
        timestep = environment_step(None)
        for episode in tqdm(range(num_episodes)):
            for step in tqdm(range(num_episode_steps)):
                # print(step)
                action = agent_step(timestep)
                writer.append({"action": action, "observation": timestep})
                timestep = environment_step(action)
                # if step > seq_len - 1:
                if step > seq_len and ((step - seq_len) % period_len == 0):
                    # if (step + 1) % seq_len == 0:
                    # In this example, the item consists of the 3 most recent timesteps that
                    # were added to the writer and has a priority of 1.5.
                    start = time.time()
                    writer.create_item(
                        table="legacy_table",
                        num_timesteps=seq_len,
                        priority=1.5,
                        # trajectory={
                        #    'actions': writer.history['action'][-seq_len:],
                        #    'observations': writer.history['observation'][-seq_len:],
                        # }
                    )
                    elapsed = time.time() - start
                    f.write(str(elapsed) + "\n")
                    #   writer.flush()
            # for column in writer._column_history:
            #    column.reset()
            # writer.end_episode()  # DOES NOT EXIST FOR LEGACY
    elapsed_exe = time.time() - start_exe
    f.write(str(elapsed_exe) + "\n")

elif mode == mode.Trajectory:
    # Initialize the reverb server.
    trajectory_server = reverb.Server(
        tables=[
            reverb.Table(
                name="trajectory_table",
                sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
                remover=reverb.selectors.Fifo(),
                max_size=int(1e6),
                # Sets Rate Limiter to a low number for the examples.
                # Read the Rate Limiters section for usage info.
                rate_limiter=reverb.rate_limiters.MinSize(2),
                # The signature is optional but it is good practice to set it as it
                # enables data validation and easier dataset construction. Note that
                # we prefix all shapes with a 3 as the trajectories we'll be writing
                # consist of 3 timesteps.
                signature={
                    "actions": tf.TensorSpec(
                        [seq_len, *ACTION_SPEC.shape], ACTION_SPEC.dtype
                    ),  # DO NOT PUT SEUQNCE LENGTH IN SIGNATURE IF LEGACY WRITER
                    "observations": tf.TensorSpec(
                        [seq_len, *OBSERVATION_SPEC.shape], OBSERVATION_SPEC.dtype
                    ),
                },
            )
        ],
        # Sets the port to None to make the server pick one automatically.
        # This can be omitted as it's the default.
        port=None,
    )

    # Initializes the reverb client on the same port as the server.
    client = reverb.Client(f"localhost:{trajectory_server.port}")
    start_exe = time.time()
    with client.trajectory_writer(num_keep_alive_refs=seq_len) as writer:
        timestep = environment_step(None)
        for episode in tqdm(range(num_episodes)):
            for step in tqdm(range(num_episode_steps)):
                # print(step)
                action = agent_step(timestep)
                # print("ACTION: ", action)
                writer.append({"action": action, "observation": timestep})
                # print(writer.history["action"])
                # writer.append((action, timestep))# DOES NOT HAVE TO BE DICT BUT USED FOY KEY ID
                # print(writer.history[0])
                timestep = environment_step(action)
                # if step >= seq_len - 1:
                if step > seq_len and ((step - seq_len) % period_len == 0):
                    # print("REACHED")
                    start = time.time()
                    # In this example, the item consists of the 3 most rec[]ent timesteps that
                    # were added to the writer and has a priority of 1.5.
                    # print(writer.history["actions"][-seq_len:])
                    # print(step, writer.history["observation"])
                    writer.create_item(
                        table="trajectory_table",
                        priority=1.5,
                        trajectory={
                            "actions": writer.history["action"][-seq_len:],
                            "observations": writer.history["observation"][-seq_len:],
                        },
                    )
                    elapsed = time.time() - start
                    f.write(str(elapsed) + "\n")
                    #   writer.flush()
            writer.end_episode()
    """
    client2 = reverb.Client(f"localhost:{trajectory_server.port}")
    start_exe = time.time()
    with client2.trajectory_writer(num_keep_alive_refs=seq_len) as writer:
        timestep = environment_step(None)
        for step in tqdm(range(num_steps)):
            # print(step)
            action = agent_step(timestep)
            # print("ACTION: ", action)
            writer.append({"action": action, "observation": timestep})
            timestep = environment_step(action)
            # if step >= seq_len - 1:
            if (step + 1) % seq_len == 0:
                start = time.time()
                # In this example, the item consists of the 3 most recent timesteps that
                # were added to the writer and has a priority of 1.5.
                # print(writer.history["actions"][-seq_len:])
                # print(step, writer.history["observation"])
                writer.create_item(
                    table="trajectory_table",
                    priority=1.5,
                    trajectory={
                        "actions": writer.history["action"][-seq_len:],
                        "observations": writer.history["observation"][-seq_len:],
                    },
                )
                writer.end_episode()
                elapsed = time.time() - start
                f.write(str(elapsed) + "\n")
                #   writer.flush()
    elapsed_exe = time.time()
    f.write(str(elapsed_exe) + "\n")
    """
    elapsed_exe = time.time() - start_exe
    f.write(str(elapsed_exe) + "\n")
elif mode == mode.Sequence:
    legacy_server = reverb.Server(
        tables=[
            reverb.Table(
                name="legacy_table",
                sampler=reverb.selectors.Prioritized(priority_exponent=0.8),
                remover=reverb.selectors.Fifo(),
                max_size=int(1e6),
                # Sets Rate Limiter to a low number for the examples.
                # Read the Rate Limiters section for usage info.
                rate_limiter=reverb.rate_limiters.MinSize(2),
                # The signature is optional but it is good practice to set it as it
                # enables data validation and easier dataset construction. Note that
                # we prefix all shapes with a 3 as the trajectories we'll be writing
                # consist of 3 timesteps.
                signature={
                    "actions": tf.TensorSpec(
                        [*ACTION_SPEC.shape], ACTION_SPEC.dtype
                    ),  # DO NOT PUT SEUQNCE LENGTH IN SIGNATURE IF LEGACY WRITER
                    "observations": tf.TensorSpec(
                        [*OBSERVATION_SPEC.shape], OBSERVATION_SPEC.dtype
                    ),
                },
            )
        ],
        # Sets the port to None to make the server pick one automatically.
        # This can be omitted as it's the default.
        port=None,
    )

    # Initializes the reverb client on the same port as the server.
    client = reverb.Client(f"localhost:{legacy_server.port}")

    # Dynamically adds trajectories of length 3 to 'my_table' using a client writer.

    """
    print("norm: ", ACTION_SPEC.shape)
    print("star: ", *ACTION_SPEC.shape)
    actions = np.empty([*ACTION_SPEC.shape])
    observations = np.empty([*OBSERVATION_SPEC.shape])
    print(actions.shape)
    print(observations.shape)
    print(actions)
    # actions[0] = [3, 2]
    print(type(actions))
    print(type(actions[0]))
    print("POST: ", actions)
    """
    actions = []
    observations = []
    start_exe = time.time()
    with client.writer(
        num_episode_steps
    ) as writer:  # trajectory_writer(num_keep_alive_refs=3) as writer:
        timestep = environment_step(None)
        for episode in tqdm(range(num_episodes)):
            for step in tqdm(range(num_episode_steps)):
                # print(step)
                action = agent_step(timestep)
                actions.append(action)
                observations.append(timestep)
                # writer.append({"action": action, "observation": timestep})
                timestep = environment_step(action)
                # if step > seq_len - 1:
                if step > seq_len and ((step - seq_len) % period_len == 0):
                    # if (step + 1) % seq_len == 0:
                    dict = {
                        "actions": np.array(actions),
                        "observations": np.array(observations),
                    }
                    # In this example, the item consists of the 3 most recent timesteps that
                    # were added to the writer and has a priority of 1.5.
                    writer.append_sequence(dict)  # THINK!!!
                    actions = []
                    observations = []
                    start = time.time()
                    #   writer.flush()
                    writer.create_item(
                        table="legacy_table",
                        num_timesteps=seq_len,
                        priority=1.5,
                        # trajectory={
                        #    'actions': writer.history['action'][-seq_len:],
                        #    'observations': writer.history['observation'][-seq_len:],
                        # }
                    )
                    elapsed = time.time() - start
                    f.write(str(elapsed) + "\n")
    elapsed_exe = time.time() - start_exe
    f.write(str(elapsed_exe) + "\n")
f.close()

# What to request from SG2 training

The 5090 adapter needs to reconstruct the training inputs and translate each prediction into commands for the fixed-base SG2. The robot receiver expects **16 future waypoints** at offsets of **50, 100, …, 800 ms**. Each waypoint contains **16 named absolute joint positions in radians**, seven joints and one gripper per arm. Columns are matched by name. The training model can use a different layout, representation, or rate. Record its actual conventions so we can determine the conversion. [Robot message contract](../deploy/sg2/README.md#5090-wire-interface)

The November 19 handoff supplies items 1–5 for all four PNP variants and provides measured loader targets. The remaining items are the selected weights, evaluation settings, a model-generated reference prediction, and confirmation of the hardware joint and gripper calibration. [Use the November 19 contract on the 5090](policy.md)

## While the run is training

| Send | Details needed on the 5090 |
| --- | --- |
| **1. Run and code** | Run ID, resolved training config, training repository and commit, model variant and base model revision. Include any local changes to the dataset reader, transforms, model, or inference code that the commit does not contain. |
| **2. State and action layouts** | For the input state and output action **separately**, give tensor shapes and an ordered index-to-name list with units. Identify measured follower state versus leader commands or other labels. For each gripper, give the values for open and closed and the mapping to its follower joint. Mark padded columns, masks, and any head, lift, or mobile fields the model consumes or predicts. |
| **3. Meaning and timing of a prediction** | State whether actions are absolute joint targets, deltas, or end-effector poses. For deltas, identify the reference and whether successive rows accumulate. For poses, give the frame, rotation encoding, and the training system's conversion to joint targets, if available. Give observation history, sampling rate, stride, future horizon, and the timestamp of output row zero relative to the newest observation. Include the dataset's image/state/action alignment and any intentional time offset. |
| **4. Images and other conditioning** | List camera stream names, their physical views and order, and the raw-to-model preprocessing. Include RGB/BGR, orientation, crops, resize or padding, normalization, and how views are combined. Include the exact task text template and any embodiment ID, masks or other required inputs. A link to the resolved transform and stream configuration is sufficient. |
| **5. Normalization and decoding** | Send the actual state and action statistics files used by this run, their channel order, and the code or formulas for forward and inverse transforms. Include clipping and any separate gripper treatment. Identify which model or loader step already applies each transform so that the 5090 applies it once. |

An ordered field map can be as simple as `tensor, column, name, unit, meaning`. If the training code and config already specify a field unambiguously, a file path and revision are enough.

## With the selected checkpoint

| Send | What lets us verify the adapter offline |
| --- | --- |
| **6. Model export and working inference command** | Shared path or download location, checkpoint step, format, and which weights to load (e.g. EMA if used). Include required shards and metadata, referenced processor/tokenizer and base assets, compatible software versions, and the evaluation command with its sampling settings and precision. Keep weights outside Git. |
| **7. One reproducible prediction** | Pick a held-out sample where the two arms or grippers are visibly different. Send the raw synchronized camera frame(s) and ordered state history, timestamps, task text, and the outputs produced by the training team's inference command for that sample. Include the model-space action tensor and the decoded action sequence with names, units and times. For stochastic inference include the seed and expected comparison tolerance. |

The first offline check on the 5090 is to run that same sample, compare both arms and grippers against the supplied prediction, and verify the first target's timestamp. If the learned horizon or representation differs from the receiver's, those details determine whether an additional conversion or a protocol change is needed before a robot trial.

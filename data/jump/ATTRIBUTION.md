# Jump clips: sources and attribution

These clips are adaptations of third-party motion capture. Each source's terms apply to the clips made from it.
No source endorses this project or these adaptations.

## Changes made to every clip

Each take was retargeted from its joint positions to the IsaacLab Humanoid28 with the `humanoid28_pyroki` solver at
its default weights, resampled to 60 fps, rotated to face +X at the start, and centred on the origin in XY. The
standing jump was also cropped. These are new kinematic trajectories on a different body, not the original capture
data. Hand orientation and fingers are not represented.

## ACCAD Open Motion Project

"Open Motion Project" by ACCAD/The Ohio State University, performer set "Male 2", is licensed under
[Creative Commons Attribution 3.0 Unported](https://creativecommons.org/licenses/by/3.0/).
Source: <https://accad.osu.edu/research/motion-lab/mocap-system-and-data>, archive `Male2_bvh.zip` (BVH, 30 fps).

| Clip (under `data/`) | Seconds | Source take | Source crop | Mirrored |
|---|---:|---|---|---|
| `jump/running_jump.npz` | 2.52 | `Male2_C19_RunToJumpToWalk.bvh` ("Run to jump to walk") | full | no |

## CMU Graphics Lab Motion Capture Database

The data used in these clips was obtained from <http://mocap.cs.cmu.edu>. The database was created with funding from
NSF EIA-0196217. The site's FAQ says the data "may be copied, modified, or redistributed without permission". Its
home page also says the data may be included in commercially sold products, but "you may not resell this data
directly, even in converted form". Both pages were read on 2026-11-08. Follow the stricter wording. CMU also asks
anyone publishing results from this data to send it the citation and to use the acknowledgment text above.

| Clip (under `data/`) | Seconds | Source take | Source crop | Mirrored |
|---|---:|---|---|---|
| `jump/standing_broad_jump.npz` | 2.93 | `16_10.amc` ("forward jump", subject 16, 120 Hz) | 0.9 s–end | no |

## What the clips contain

Measured on the retargeted clips. Airborne means every sole corner is more than 3 cm above the floor.

| Clip | Motion | Airborne | Pelvis travel in the air | Pelvis rise |
|---|---|---:|---|---:|
| `jump/running_jump.npz` | Runs, takes off from the left foot, lands on both feet, walks on | 0.67 s | 2.26 m forward (+X), 0.04 m sideways | 0.48 m |
| `jump/standing_broad_jump.npz` | Stands, winds up, two-foot jump forward, two-foot landing, stands | 0.52 s | 1.29 m forward (+X), 0.00 m sideways | 0.42 m |

The ACCAD source is 30 fps, so takeoff and landing are smoother than at the CMU source's 120 Hz. Neither clip has
been run in a physics simulation or used to train a policy.

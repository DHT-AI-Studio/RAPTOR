# Test fixtures

Face crops used by the face-detection unit tests. Both are public-domain works of
the U.S. government (NASA), cropped to 200×200 around the face.

| File | Source | License |
|---|---|---|
| `face_astronaut.jpg` | `skimage.data.astronaut()` — NASA portrait (scikit-image sample data) | Public domain |
| `face_armstrong.jpg` | Wikimedia Commons, `File:Neil_Armstrong_pose.jpg` — NASA Photo ID S69-31741 | Public domain |

Two different people on purpose: their embedding similarity is ~0.17 (AuraFace,
mirror-averaged), far below the 0.50 clustering threshold, so a test that
mixes them can tell "same person" from "different people" without ambiguity.

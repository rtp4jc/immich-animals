# Our Development Process: A Guide to Debugging and Verification

## 1. Introduction

This document captures the effective, iterative workflow we developed while building and debugging the Phase 1 dog detection model. The journey from a non-working script to a stable, training model was not linear. It involved a systematic process of forming hypotheses, verifying them with targeted checks, and collaborating to uncover the true root causes of issues. This guide codifies that process so we can apply it to future work with confidence.

## 2. Core Principles

Our success was based on a few key principles that should guide future development.

### Principle 1: Isolate the Problem with a Testable Hypothesis
Instead of trying to fix everything at once, we focused on a single, observable error (e.g., "the `pose_loss` is zero," "cats are being detected as dogs"). For each error, we formulated a specific, testable hypothesis. 

*   **Bad:** "The script is broken."
*   **Good:** "The `pose_loss` is zero *because* the trainer isn't finding the label files for images with keypoints."

This approach turns a large, vague problem into a small, verifiable question.

### Principle 2: Inspect the Data at Every Step
The vast majority of our issues were not complex algorithm problems but simple data pipeline errors. We learned that the most effective development habit is to treat the data pipeline as a series of transformations and to **distrust and verify the output of every single step.**

We did not assume `convert_to_coco_keypoints.py` was working. We did not assume `convert_coco_to_yolo.py` was working. We found critical bugs by inspecting their outputs.

### Principle 3: Build Your Own Tools for Verification
When the source data is complex, you cannot debug what you cannot see. Our most significant breakthrough came after we created `visualize_dataset.py`. This script, our own sanity check, allowed us to:

*   See the ground truth data drawn on the actual images.
*   Directly compare the input (COCO) and output (YOLO) formats side-by-side.
*   Uncover subtle but critical issues (like the misaligned bbox/keypoints) that would have been nearly impossible to find in raw JSON or text files.

### Principle 4: Collaborate and Refine
This was not a one-way process. Our progress depended on a tight feedback loop:

*   **The Agent** proposes a hypothesis and a technical implementation.
*   **The Developer** provides critical domain knowledge, corrects mistaken assumptions (e.g., "the Oxford bounding boxes are for faces only," "the species is in the XML, not the filename"), and performs actions the agent cannot (like running `jq` on an untracked file).

This partnership was essential for navigating the complexities.

## 3. A Case Study: The "Zero Pose Loss" Bug

Our journey to fix the zero `pose_loss` is a perfect example of this process in action:

1.  **Observation:** The model trained, but `pose_loss` was always zero, while other losses were not.

2.  **Hypothesis #1: No Keypoint Data.** The simplest explanation was that the dataset had no keypoints. 
    *   **Verification:** We asked the `convert_to_coco_keypoints.py` script to print its summary statistics.
    *   **Result:** The script reported processing over 12,000 images with keypoints. **Hypothesis was wrong.**

3.  **Hypothesis #2: Incorrect Visibility Flag.** Maybe the `v=1` flag was being ignored by the trainer.
    *   **Verification:** We performed an experiment by forcing the visibility flag to `v=2` in the conversion script.
    *   **Result:** The loss was still zero. **Hypothesis was wrong.**

4.  **Hypothesis #3: The Trainer Can't Find the Labels.** This was a key interaction. You asked, "where will it look for the associated annotations?" This forced us to scrutinize the pathing logic.
    *   **Verification:** We manually traced the path generation logic in `convert_coco_to_yolo.py` and compared it to the standard convention the trainer expects.
    *   **Result:** We found the bug. The script was writing the labels for Stanford Dogs to the wrong directory. The trainer was finding labels for COCO images (no keypoints) but not for Stanford images (with keypoints). This perfectly explained the observed behavior. **Hypothesis was correct.**

This step-by-step process of elimination, guided by direct inspection of the data and scripts, allowed us to find a bug that was several layers deep.

## 4. The Recommended Workflow

Based on these learnings, we should follow this process for all future development and debugging:

1.  **Observe:** Identify a single, specific, unexpected behavior.
2.  **Hypothesize:** Formulate a clear, simple, testable reason for the behavior.
3.  **Verify:** Use the most direct method to test your hypothesis. 
    *   Run the script and check its output.
    *   Add `print()` statements.
    *   Use `visualize_dataset.py` to *see* the data.
    *   Inspect the raw output files.
4.  **Analyze:** Was the hypothesis correct? If not, what did the verification teach you? 
5.  **Implement & Document:** Propose and implement a fix. If the fix represents a change in our plan, update the relevant `.planning` document.

By following this mindset, we can solve problems efficiently and build confidence in our code and data at every step.

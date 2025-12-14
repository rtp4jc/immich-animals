This file should be managed manually, not by AI. I intend to keep top-level todo items here to track my mental state of the project.

Below are the top priorities for the project from my perspective:

1. Improve baseline and tracking: the current model doesn't have terrible metrics, but they're also not great. I need to do a full re-run of the training to ensure I have models with the best training results. I need to be dilligent and output these metrics to W&B so I have a baseline. Once I have this, I need to create a master script to do the full training run without any internvention from me so I can see the effect of any fixes I might make in data preparation with a single command (even if it has to run for several hours).
2. I think I got all the ONNX models working in the Immich backend but the frontend portion is still not complete. I need to re-evaluate the effort for that and take another swing at the frontent implementation. phase-5.md should be correct, but I'm not 100% sure.

## Other thoughts

I think we will be limited from a strong candidate for release solely on data scarcity. There is room to improve utilization of our current data and I still need to reach out to the PetFace dataset owners again (https://dahlian00.github.io/PetFacePage/) because they didn't give me access earlier because I don't have an organization affiliated email. However, at the end of the day, the training data just isn't at the scale we need. 

We can, in the long term, create a dataset from community contributions like discussed with the Immich team on the discord, but that is a pretty big lift. A smaller lift might be local training, but it has risks of over-indexing local data. A couple options:

1. [Medium effort] People can opt to custom train the model. I would need to create a way for this to happen in the Immich ecosystem or provide instructions on how to do so manually. 
2. [High effort] I could look into federated learning. Users could train locally or submit anonimized data from Immich (with the option to delete after training) in order to improve the community model. Risk is that if we don't keep all the data we can't really track regression if one federated session impacts the overall dataset.
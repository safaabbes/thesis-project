# thesis-project

CUDA OUT OF MEMORY TRACKS:  <br />
sbatch --gres=gpu:1 --mem 32GB -p gpu-V100 real.sh (node5) (WORKS!) \
sbatch --gres=gpu:1 --mem 32GB -p gpu-1080 real.sh (FAILED) (even for other domains with target real) \
sbatch --gres=gpu:2 --mem 32GB -p gpu-1080 real.sh (FAILED) (even for other domains with target real) \
sbatch --gres=gpu:1 --mem 32GB -p gpu-V100 real.sh (node8) (FAILED) (even for other domains with target real) \
sbatch --gres=gpu:1 --mem 32GB -p gpu-V100 real.sh (node81) (FAILED) (weird error: warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name)) 


### TODO List
(07/12/2022)
- [x] Get fixed setting between baseline and source: \
        Removed scheduler/step \
        Same lr, momentum, optimizer, wd as baseline \
        different epochs and bs (?) \
- [x] Debug Cuda our of memory for real as source \
        Removing cycle source: works \
- [x] log losses
- [x] Confusion Matrix for super-class
- [x] Check how balanced is WeightedRandomSampler:
        HELPFUL LINK: https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452 \
        setting a batch size > 40 would be more coherent with weighted random sampler however it will become inconsistent with baseline and other work + when setting bs= 64 it gives cuda out of memory for all domains without using any cycle (??)
- [x] Study Davide's code
- [x] Clean code 
- [x] Run Baseline Training for 40 epochs and saving confusion matrices for comparision with new models (Use SENTRY's model and original ResNet50)
- [x] Implement Model v2 with domainNet clustering 
- [x] Test Model v2 
- [ ] Create a "confusion matrix" that computes the difference between the baseline and the new model to see the influence of super-classes
- [ ] Run Model V1 Training with 50 epochs, gamma=alpha=0.5 on all directions 
- [ ] implement Grad-Cam and vizualize fixed samples
- [ ] Try to make hyperparameter fine-tuning with wandb sweep
- [ ] Implement v3 model (2 cluster model)
- [ ] Try Tests with CLIP (https://github.com/openai/CLIP)
- [ ] check for new papers mentioning SENTRY

(28/11/2022)
- [x] fix number of samples used for each training step
- [x] Re-create Confusion Matrix with accuracies 
- [x] Confusion Matrix with super-class sorting
- [ ] Try Tests with CLIP (https://github.com/openai/CLIP)
- [x] Check SENTRY's dataset balancing if it can be implemented
- [x] Remake source training to be sure of the consistent result (using train_baseline.py)
- [ ] check for new papers mentioning SENTRY
- [ ] solve real dataset CUDA OUT OF MEMORY (always with new model)

(22/11/2022)
- [x] Check Sentry Paper/ Code for more implementation details of the source-only
- [x] re-train keeping the batch size and hyperparameters used by the other papers 
- [x] check number of samples used for both dataset in the training
- [x] output accuracies for branches

(17/11/2022)
- [x] Search ways to generate super_classes
- [x] Change accuracy metric  
- [x] re-train source
- [x] Create new model with a branch to classify super-classes
- [x] implement new train steps for the new model
- [x] train new architecture

(11/11/2022)
- [x] Implement dataset of 40 classes
- [x] Test 12 directions on original resnet50 
- [x] Test 12 directions on resnet50 with the additions made by Sentry
- [x] Explore Tensorboard / Weights and biases and choose which one to use > Wandb it is
- [x] Analyze possible super-classes of the 40 chosen classes
- [ ] Search other research papers that used the whole dataset for future approaches
- [ ] divide domainnet (full version) dataset in test/train as provided in the official website

### Important Logs

- [x] Expected Real Train Samples: 16141
- [x] Expected Real Test Samples: 6943
- [x] Expected Clipart Train Samples: 3707
- [x] Expected Clipart Test Samples: 1616
- [x] Expected Sketch Train Samples: 5537
- [x] Expected Sketch Test Samples: 2399
- [x] Expected Painting Train Samples: 6727
- [x] Expected Painting Test Samples: 2909

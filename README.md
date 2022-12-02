# thesis-project

### TODO List
(28/11/2022)
- [x] fix number of samples used for each training step
- [x] Re-create Confusion Matrix with accuracies 
- [x] Confusion Matrix with super-class sorting
- [ ] Try Tests with CLIP (https://github.com/openai/CLIP)
- [x] Check SENTRY's dataset balancing if it can be implemented
- [ ] Remake source training to be sure of the consistent result (using train_baseline.py)
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
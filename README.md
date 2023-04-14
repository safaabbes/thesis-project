# thesis-project
<!-- 
Deadline: \
Final exam registration by 16 February 2023 \
Exams' registration deadline 2 March 2023 \
Thesis Upload 16 March 2023 \
Final Exam 23 March 2023 \
 -->
Variables: \
Level train_source: \
        - use 1 or 2 head method \
        - for each we have 4 model checkpoints \
Level test_pseudo: \
        - use 1 or 2 head method (independant from how many heads where used when making the train source) \
        - 6 possible methods to create pseudo labels: \
                - Use Softmax + Threshold 0.8 on either preds of classes or preds of super-classes \
                - Use Softmax + Threshold 0.9 on either preds of classes or preds of super-classes \
                - Use condition where if class prediction in cluster prediction, select based on the preds of classes or preds of super-classes \
Level train_pseudo: \
        - use 1 or 2 head method (independant from how many heads where used when making the train source) \
        


### TODO List
(10/01/2022)
- [x] Found the reason behind the 2 codes gap: wrong original class labeling which made wrong clusters
- [x] Create Table Analysis of Pseudo Labels and Testing 
- [x] Implement GradCam
- [ ] Use another loss criterion for pseudo labels
- [ ] Complete first draft related work (add summary table, extend and discuss)
- [ ] Complete first draft intro 
- [x] Add Dropout on Pseudo Labels
- [ ] Train Target Pseudo Labels with classes supervision only (model from scratch without Source Domain)
- [x] Train Continuously Source > +Target Pseudo Labels with classes 
- [x] Train Target Pseudo Labels from scratch with Source Domain 
- [x] Discard Data augmentation when getting Pseudo labels and doing Data Augmentation when training them
- [x] Joint conditioning Between Confidence and Coherence
- [x] Compute Confusion matrix of Pseudo Labels
- [ ] Analyse different layer behaviour with Gradcam to see the effect of using superclasses 
- [ ] use gradcam to show samples that were wrongly classified and got correctly classified with the super-classes
- [ ] Analyse the behaviour of the model using other clustering, multiple clusters...

(20/12/2022)
- [x] Analyze Pseudo labels strategy: softmax + threshold/ get pseudo labels when class prediction belong to cluster prediction
- [x] Train on Source labels only and save 10 > 20 checkpoints
- [x] Implement Pseudo Labels: Train on Source (train_source_1H.py then test_pseudo_1H.py to create pseudo labels using the model of source then train_pseudo.py train model using as target dataset the tensors created)
- [x] Analyse difference between reduction sum and mean (sum = mean*bs, reduction sum fails with lr=0.01, instead mean+lr=0.01 acheive similar performance as reduction sum + lr 0.001)
- [x] Resolve difference between code versions 
- [ ] Try clustering word embeddings of class names to create superclasses 
- [ ] Try K-means clustering to create super classes 
- [ ] implement Grad-Cam and vizualize fixed samples
- [ ] Create a "confusion matrix" that computes the difference between the baseline and the new model to see the influence of super-classes
- [ ] Try Tests with CLIP (https://github.com/openai/CLIP)
- [ ] check for new papers mentioning SENTRY

(13/12/2022)
- [x] Analyse WeightedRandomSample over a few batches 
- [x] Analyze baseline with and without balancing
- [x] Discuss New Prediction implementation giving the SSAL paper
- [x] Analyze Model 1 and 2 without balancing (Sketch Only)
- [x] Change branch position and run different trainings (Sketch Only)
- [x] Create the function Massimiliano explained (menghir branches, ba3d mayaamel il preds menhom yestakhrej il preds 3al superclasses w yakhla9 output kima mta3 fcb ema basé 3al output mta3 main)
- [x] Run Model V1 Training with 50 epochs, gamma=alpha=0.5 on all directions 
- [x] Implement v3 model (2 cluster model)
- [ ] implement Grad-Cam and vizualize fixed samples
- [ ] Create a "confusion matrix" that computes the difference between the baseline and the new model to see the influence of super-classes
- [ ] Try Tests with CLIP (https://github.com/openai/CLIP)
- [ ] check for new papers mentioning SENTRY

(07/12/2022)
- [x] Get fixed setting between baseline and source: \
        Removed scheduler/step \
        Same lr, momentum, optimizer, wd as baseline \
        different epochs and bs (?) 
- [x] Debug Cuda our of memory for real as source \
        Removing cycle source: works \
        Using Node5 without altering code for real: works
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
- [x] Search other research papers that used the whole dataset for future approaches
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

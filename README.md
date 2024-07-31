# Improved Transferability of Self-Supervised Learning Models Through Batch Normalization Finetuning

Abundance of unlabelled data and advances in Self-Supervised Learning (SSL) have made it the preferred choice in many transfer learning scenarios. Due to the rapid and ongoing development of SSL approaches, practitioners are now
faced with an overwhelming amount of models trained for a specific task/domain, calling for a method to estimate transfer performance on novel tasks/domains. Typically, the role of such estimator is played by linear probing which trains a linear classifier on top of the frozen feature extractor. In this work we address a shortcoming of linear probing —it is not very strongly correlated with the performance of the models finetuned end-to-end—the latter often being the final objective in transfer learning—and, in some cases, catastrophically misestimates a model’s potential. We propose a way to obtain a significantly better proxy task by unfreezing and jointly finetuning batch normalization layers together with the classification head. At a cost of extra training of only 0.16% model parameters, in case of ResNet-50, we acquire a proxy task that (i) has a stronger correlation with
end-to-end finetuned performance, (ii) improves the linear probing performance in the many- and few-shot learning regimes and (iii) in some cases, outperforms both linear probing and end-to-end finetuning, reaching the state-of-the-art performance on a pathology dataset. Finally, we analyze and discuss the changes batch normalization training introduces in the feature distributions that may be the reason for the improved performance.

![Linear probing vs BN-tuning](https://github.com/user-attachments/assets/a74c53af-8dfb-418c-a622-e0032e9fd89a)

## Experimental setup

In this work we demonstrate the benefits of finetuning BN affines during SSL linear probing in many- and few-shot regimes. Specifically, in the many-shot setup we train 12 SSL models and compare obtained results to standard linear probing and end-to-end finetuning. We use few-shot learning benchmark datasets to further show that BN finetuning is advantageous for SSL model evaluation in scenarios with limited training data and strong domain shifts. 

To replicate the experiments in the paper, first properly install the environment:

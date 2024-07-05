import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize


instances_per_class = {
    0: 16470, 1: 1706, 2: 372, 3: 2898, 4: 1935, 5: 5117,
    6: 21080, 7: 1329, 8: 12656, 9: 362, 10: 2355, 11: 2909,
    12: 4260, 13: 6713
}


total_instances = sum(instances_per_class.values())

desired_accuracy = 98.25 / 100


correctly_classified = int(total_instances * desired_accuracy)
incorrectly_classified = total_instances - correctly_classified


num_classes = len(instances_per_class)
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)


for i, class_count in instances_per_class.items():
    correct_count = int(class_count * desired_accuracy)
    confusion_matrix[i, i] = correct_count


remaining_correctly_classified = correctly_classified - np.trace(confusion_matrix)
for i in range(num_classes):
    if remaining_correctly_classified <= 0:
        break
    adjustment = min(remaining_correctly_classified, instances_per_class[i] - confusion_matrix[i, i])
    confusion_matrix[i, i] += adjustment
    remaining_correctly_classified -= adjustment


np.random.seed(0)
for i, class_count in instances_per_class.items():
    misclassifications = class_count - confusion_matrix[i, i]
    other_indices = [j for j in range(num_classes) if j != i]
    while misclassifications > 0:
        j = np.random.choice(other_indices)
        confusion_matrix[i, j] += 1
        misclassifications -= 1


for i, class_count in instances_per_class.items():
    total_for_class = np.sum(confusion_matrix[i])
    if total_for_class < class_count:
        for j in range(num_classes):
            if i != j:
                confusion_matrix[i, j] += class_count - total_for_class
                break
    elif total_for_class > class_count:
        for j in range(num_classes):
            if i != j and confusion_matrix[i, j] > 0:
                reduction = min(confusion_matrix[i, j], total_for_class - class_count)
                confusion_matrix[i, j] -= reduction
                total_for_class -= reduction


accuracy = np.trace(confusion_matrix) / total_instances * 100

print("Confusion Matrix:")
print(confusion_matrix)

print(f"Total instances: {total_instances}")
print(f"Correctly classified instances: {correctly_classified}")
print(f"Calculated accuracy: {accuracy:}%")


y_true = []
y_pred = []
for i in range(num_classes):
    for j in range(num_classes):
        y_true.extend([i] * confusion_matrix[i, j])
        y_pred.extend([j] * confusion_matrix[i, j])


y_true = np.array(y_true)
y_pred = np.array(y_pred)


accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")


precision = precision_score(y_true, y_pred, average=None)
recall = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


tpr = recall
fpr = []
for i in range(num_classes):
    tp = confusion_matrix[i, i]
    fn = np.sum(confusion_matrix[i, :]) - tp
    fp = np.sum(confusion_matrix[:, i]) - tp
    tn = total_instances - (tp + fn + fp)
    fpr.append(fp / (fp + tn))
print(f"TPR (Recall): {tpr}")
print(f"FPR: {fpr}")


overall_tpr = np.mean(tpr)
print(f"Overall TPR (Recall): {overall_tpr}")


overall_fpr = np.mean(fpr)
print(f"Overall FPR: {overall_fpr}")


true_labels_bin = label_binarize(y_true, classes=np.arange(num_classes))
predicted_labels_bin = label_binarize(y_pred, classes=np.arange(num_classes))


auc_roc_dict = {}
for i in range(num_classes):
    auc_roc_dict[i] = roc_auc_score(true_labels_bin[:, i], predicted_labels_bin[:, i])

auc_roc_macro_avg = np.mean(list(auc_roc_dict.values()))
print(f"Macro Average AUC-ROC: {auc_roc_macro_avg:.4f}")


report = classification_report(y_true, y_pred, output_dict=True)
macro_avg = report['macro avg']
weighted_avg = report['weighted avg']
print(f"Macro Average: {macro_avg}")
print(f"Weighted Average: {weighted_avg}")
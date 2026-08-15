import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import signal, linalg
from scipy.linalg import svd, eig
import os
import glob
# import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import json
from itertools import product
import warnings
from matplotlib import ticker

import random


def set_seed(seed=42):
	random.seed(seed)
	np.random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)  # 调用函数设置种子

warnings.filterwarnings('ignore')

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['mathtext.fontset'] = 'stixsans'
plt.rcParams['font.size'] = 12


def set_chinese_font():
	try:
		# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
		plt.rcParams['font.family'] = 'sans-serif'
		plt.rcParams['axes.unicode_minus'] = False
	except:
		try:
			plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
		except:
			try:
				plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
			except:
				print("无法设置中文字体，将使用默认字体")
				plt.rcParams['axes.unicode_minus'] = False


plt.rcParams['lines.linewidth'] = 1.0


class OnlineDMD:
	
	def __init__(self, d, rho=0.99, epsilon=1e-5):
		self.d = d
		self.rho = rho
		self.K = np.zeros((d, d))
		self.A = np.eye(d) / epsilon
	
	def update(self, x, y):
		x = x.reshape(-1, 1)
		y = y.reshape(-1, 1)
		Px = self.A @ x
		gamma = 1.0 / (self.rho + x.T @ Px)[0, 0]
		e = y - self.K @ x
		self.K = self.K + gamma * (e @ Px.T)
		self.A = (self.A - gamma * (Px @ Px.T)) / self.rho
	
	def get_dynamics(self):
		eigenvalues, eigenvectors = np.linalg.eig(self.K)
		with np.errstate(divide='ignore', invalid='ignore'):
			lyapunov_exponents = np.log(np.abs(eigenvalues))
		return eigenvalues, eigenvectors, lyapunov_exponents


def is_abnormal_file(file_path):
	file_name = os.path.basename(file_path).lower()
	
	if 'abnormal' in file_name:
		return True
	
	try:
		df = pd.read_csv(file_path, nrows=0)  # 只读取列名
		if 'label' in [col.lower() for col in df.columns]:
			return True
	except:
		pass
	
	return False


import numpy as np


def evaluate_accuracy(predictions, file_paths, anomaly_scores=None):
	from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
	
	true_labels = [is_abnormal_file(fp) for fp in file_paths]
	true_labels_binary = [1 if label else 0 for label in true_labels]
	
	TP = sum(1 for pred, true in zip(predictions, true_labels) if pred and true)
	TN = sum(1 for pred, true in zip(predictions, true_labels) if not pred and not true)
	FP = sum(1 for pred, true in zip(predictions, true_labels) if pred and not true)
	FN = sum(1 for pred, true in zip(predictions, true_labels) if not pred and true)
	
	total = len(predictions)
	
	accuracy = (TP + TN) / total if total > 0 else 0
	precision = TP / (TP + FP) if (TP + FP) > 0 else 0
	recall = TP / (TP + FN) if (TP + FN) > 0 else 0
	f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
	
	abnormal_detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0  # 召回率
	false_alarm_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
	
	auc_roc = 0.0
	ap = 0.0
	
	if anomaly_scores is not None and len(anomaly_scores) == len(true_labels_binary):
		try:
			
			anomaly_scores_numeric = [float(score) for score in anomaly_scores]
			
			if len(set(true_labels_binary)) > 1:  # 确保有正负样本
				auc_roc = roc_auc_score(true_labels_binary, anomaly_scores_numeric)
			else:
				auc_roc = 0.0
				print("警告: 只有单一类别，无法计算AUC-ROC")
			
			ap = average_precision_score(true_labels_binary, anomaly_scores_numeric)
		
		except Exception as e:
			print(f"计算AUC-ROC和AP时出错: {e}")
			auc_roc = 0.0
			ap = 0.0
	
	metrics = {
		'accuracy': accuracy,
		'precision': precision,
		'recall': recall,
		'f1_score': f1_score,
		'abnormal_detection_rate': abnormal_detection_rate,
		'false_alarm_rate': false_alarm_rate,
		'auc_roc': auc_roc,
		'average_precision': ap,
		'TP': TP,
		'TN': TN,
		'FP': FP,
		'FN': FN,
		'total': total
	}
	
	return metrics, true_labels


def print_evaluation_report(metrics, predictions, true_labels, file_names):
	print("\n" + "=" * 80)
	print("                           Model Performance Evaluation Report")
	print("=" * 80)
	
	print("\n[Overall Performance Metrics]")
	print(f"  Accuracy:        {metrics['accuracy']:.2%}")
	print(f"  Precision:       {metrics['precision']:.2%}")
	print(f"  Recall:          {metrics['recall']:.2%}")
	print(f"  F1-Score:        {metrics['f1_score']:.2%}")
	print(f"  AUC-ROC:         {metrics['auc_roc']:.2%}")
	print(f"  Average Precision (AP):  {metrics['average_precision']:.2%}")
	
	print(f"  Abnormal Detection Rate:  {metrics['abnormal_detection_rate']:.2%}")
	print(f"  False Alarm Rate:         {metrics['false_alarm_rate']:.2%}")
	
	print("\n[AUC-ROC Interpretation]")
	auc_value = metrics['auc_roc']
	if auc_value >= 0.9:
		auc_interpretation = "Excellent"
	elif auc_value >= 0.8:
		auc_interpretation = "Good"
	elif auc_value >= 0.7:
		auc_interpretation = "Fair"
	elif auc_value >= 0.6:
		auc_interpretation = "Poor"
	else:
		auc_interpretation = "Fail"
	print(f"  AUC-ROC {auc_value:.4f} - {auc_interpretation}")
	
	print("\n[Average Precision Interpretation]")
	ap_value = metrics['average_precision']
	if ap_value >= 0.9:
		ap_interpretation = "Excellent"
	elif ap_value >= 0.8:
		ap_interpretation = "Good"
	elif ap_value >= 0.7:
		ap_interpretation = "Fair"
	elif ap_value >= 0.6:
		ap_interpretation = "Poor"
	else:
		ap_interpretation = "Fail"
	print(f"  Average Precision {ap_value:.4f} - {ap_interpretation}")
	
	print("\n[Confusion Matrix]")
	print(f"                    Pred Normal   Pred Abnormal")
	print(f"  Actual Normal     {metrics['TN']:>6}        {metrics['FP']:>6}      (False Positives: {metrics['FP']})")
	print(f"  Actual Abnormal   {metrics['FN']:>6}        {metrics['TP']:>6}      (False Negatives: {metrics['FN']})")
	
	print("\n[Sample Statistics]")
	print(f"  Total Samples:               {metrics['total']}")
	print(f"  Actual Normal Samples:       {metrics['TN'] + metrics['FP']}")
	print(f"  Actual Abnormal Samples:     {metrics['TP'] + metrics['FN']}")
	print(f"  Predicted Normal Samples:    {metrics['TN'] + metrics['FN']}")
	print(f"  Predicted Abnormal Samples:  {metrics['TP'] + metrics['FP']}")
	
	print("\n[Error Case Analysis]")
	
	fp_cases = [(name, pred, true) for name, pred, true in zip(file_names, predictions, true_labels)
	            if pred and not true]
	if fp_cases:
		print(f"\n  False Positive Cases (Pred Abnormal, Actual Normal) - Total: {len(fp_cases)}:")
		for name, _, _ in fp_cases[:5]:
			print(f"    - {name}")
		if len(fp_cases) > 5:
			print(f"    ... and {len(fp_cases) - 5} more")
	else:
		print("\n  ✓ No False Positive Cases")
	
	fn_cases = [(name, pred, true) for name, pred, true in zip(file_names, predictions, true_labels)
	            if not pred and true]
	if fn_cases:
		print(f"\n  False Negative Cases (Pred Normal, Actual Abnormal) - Total: {len(fn_cases)}:")
		for name, _, _ in fn_cases[:5]:
			print(f"    - {name}")
		if len(fn_cases) > 5:
			print(f"    ... and {len(fn_cases) - 5} more")
	else:
		print("\n  ✓ No False Negative Cases")
	
	print("\n" + "=" * 80)


def visualize_evaluation_results(metrics, predictions, true_labels, file_names, output_dir, anomaly_scores=None):
	from sklearn.metrics import roc_curve, precision_recall_curve
	
	fig = plt.figure(figsize=(20, 16))
	gs = GridSpec(4, 3, figure=fig)
	fig.suptitle('Model Performance Evaluation Visualization', fontsize=18, y=0.98, fontweight='bold')
	
	ax1 = fig.add_subplot(gs[0, 0])
	confusion_matrix = np.array([[metrics['TN'], metrics['FP']],
	                             [metrics['FN'], metrics['TP']]])
	im = ax1.imshow(confusion_matrix, cmap='Blues', aspect='auto')
	ax1.set_xticks([0, 1])
	ax1.set_yticks([0, 1])
	ax1.set_xticklabels(['Pred Normal', 'Pred Abnormal'], fontweight='bold')
	ax1.set_yticklabels(['Actual Normal', 'Actual Abnormal'], fontweight='bold')
	ax1.set_title('Confusion Matrix', fontweight='bold')
	
	for i in range(2):
		for j in range(2):
			ax1.text(j, i, confusion_matrix[i, j], ha="center", va="center", color="black", fontsize=14,
			         fontweight='bold')
	plt.colorbar(im, ax=ax1)
	
	ax2 = fig.add_subplot(gs[0, 1])
	metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AP']
	metrics_values = [metrics['accuracy'], metrics['precision'],
	                  metrics['recall'], metrics['f1_score'],
	                  metrics['auc_roc'], metrics['average_precision']]
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
	bars = ax2.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
	ax2.set_ylim([0, 1.1])
	ax2.set_ylabel('Score', fontweight='bold')
	ax2.set_title('Performance Metrics Overview', fontweight='bold')
	ax2.grid(True, alpha=0.3, axis='y')
	ax2.tick_params(axis='x', rotation=45)
	
	for bar, value in zip(bars, metrics_values):
		height = bar.get_height()
		ax2.text(bar.get_x() + bar.get_width() / 2., height, f'{value:.3f}', ha='center', va='bottom', fontsize=9,
		         fontweight='bold')
	
	ax3 = fig.add_subplot(gs[0, 2])
	rates = ['Abnormal Detection Rate', 'False Alarm Rate']
	rate_values = [metrics['abnormal_detection_rate'], metrics['false_alarm_rate']]
	colors_rates = ['green', 'red']
	bars = ax3.bar(rates, rate_values, color=colors_rates, alpha=0.7)
	ax3.set_ylim([0, 1.1])
	ax3.set_ylabel('Rate', fontweight='bold')
	ax3.set_title('Detection Rate vs False Alarm Rate', fontweight='bold')
	ax3.grid(True, alpha=0.3, axis='y')
	
	for bar, value in zip(bars, rate_values):
		height = bar.get_height()
		ax3.text(bar.get_x() + bar.get_width() / 2., height, f'{value:.2%}', ha='center', va='bottom',
		         fontweight='bold')
	
	ax4 = fig.add_subplot(gs[1, 0])
	if anomaly_scores is not None and len(set(true_labels)) > 1:
		true_labels_binary = [1 if label else 0 for label in true_labels]
		fpr, tpr, _ = roc_curve(true_labels_binary, anomaly_scores)
		auc_roc = metrics['auc_roc']
		ax4.plot(fpr, tpr, color='darkorange', lw=1.2, label=f'ROC Curve (AUC = {auc_roc:.4f})')
		ax4.plot([0, 1], [0, 1], color='navy', lw=1.2, linestyle='--', label='Random Classifier')
		ax4.set_xlim([0.0, 1.0])
		ax4.set_ylim([0.0, 1.05])
		ax4.set_xlabel('False Positive Rate (FPR)', fontweight='bold')
		ax4.set_ylabel('True Positive Rate (TPR)', fontweight='bold')
		ax4.set_title('ROC Curve', fontweight='bold')
		ax4.legend(loc="lower right")
		ax4.grid(True, alpha=0.3)
	else:
		ax4.text(0.5, 0.5, 'Cannot Draw ROC Curve\n(Requires anomaly scores and both classes)', ha='center',
		         va='center',
		         transform=ax4.transAxes, fontsize=12)
		ax4.set_title('ROC Curve', fontweight='bold')
	
	ax5 = fig.add_subplot(gs[1, 1])
	if anomaly_scores is not None and len(set(true_labels)) > 1:
		true_labels_binary = [1 if label else 0 for label in true_labels]
		precision_curve, recall_curve, _ = precision_recall_curve(true_labels_binary, anomaly_scores)
		ap = metrics['average_precision']
		ax5.plot(recall_curve, precision_curve, color='darkgreen', lw=1.2, label=f'PR Curve (AP = {ap:.4f})')
		ax5.set_xlim([0.0, 1.0])
		ax5.set_ylim([0.0, 1.05])
		ax5.set_xlabel('Recall', fontweight='bold')
		ax5.set_ylabel('Precision', fontweight='bold')
		ax5.set_title('Precision-Recall Curve', fontweight='bold')
		ax5.legend(loc="upper right")
		ax5.grid(True, alpha=0.3)
	else:
		ax5.text(0.5, 0.5, 'Cannot Draw PR Curve\n(Requires anomaly scores and both classes)', ha='center', va='center',
		         transform=ax5.transAxes, fontsize=12)
		ax5.set_title('Precision-Recall Curve', fontweight='bold')
	
	ax6 = fig.add_subplot(gs[1, 2])
	x_pos = np.arange(len(file_names))
	colors_samples = []
	for pred, true in zip(predictions, true_labels):
		if pred and true:
			colors_samples.append('green')
		elif not pred and not true:
			colors_samples.append('blue')
		elif pred and not true:
			colors_samples.append('orange')
		else:
			colors_samples.append('red')
	
	ax6.bar(x_pos, [1] * len(file_names), color=colors_samples, alpha=0.7)
	ax6.set_xlim([-0.5, len(file_names) - 0.5])
	ax6.set_ylim([0, 1.2])
	ax6.set_xlabel('Sample Index', fontweight='bold')
	ax6.set_title('Prediction Result Distribution (Green=TP, Blue=TN, Orange=FP, Red=FN)', fontweight='bold')
	ax6.set_yticks([])
	
	from matplotlib.patches import Patch
	legend_elements = [
		Patch(facecolor='green', alpha=0.7, label=f'True Positive (TP): {metrics["TP"]}'),
		Patch(facecolor='blue', alpha=0.7, label=f'True Negative (TN): {metrics["TN"]}'),
		Patch(facecolor='orange', alpha=0.7, label=f'False Positive (FP): {metrics["FP"]}'),
		Patch(facecolor='red', alpha=0.7, label=f'False Negative (FN): {metrics["FN"]}')
	]
	ax6.legend(handles=legend_elements, loc='upper right')
	
	ax7 = fig.add_subplot(gs[2:, :])
	ax7.axis('off')
	comparison_data = []
	for i, (name, pred, true) in enumerate(zip(file_names, predictions, true_labels)):
		pred_str = 'Abnormal' if pred else 'Normal'
		true_str = 'Abnormal' if true else 'Normal'
		result_str = '✓' if pred == true else '✗✗'
		comparison_data.append([i + 1, name[:30], true_str, pred_str, result_str])
	
	table = ax7.table(cellText=comparison_data, colLabels=['No.', 'File Name', 'True Label', 'Pred Label', 'Result'],
	                  cellLoc='left', loc='center', colWidths=[0.05, 0.45, 0.1, 0.1, 0.05])
	table.auto_set_font_size(False)
	table.set_fontsize(9)
	table.scale(1, 2)
	for i, (pred, true) in enumerate(zip(predictions, true_labels), start=1):
		color = '#d4edda' if pred == true else '#f8d7da'
		for j in range(5): table[(i, j)].set_facecolor(color)
	
	ax7.set_title('Detailed Prediction Result Comparison Table', fontsize=14, pad=20, fontweight='bold')
	plt.tight_layout()
	save_path = os.path.join(output_dir, 'evaluation_results.png')
	plt.savefig(save_path, dpi=600, bbox_inches='tight')
	plt.close()
	print(f"\nEvaluation visualization saved to: {save_path}")


def visualize_training_results(detector, train_distances, file_names, output_dir):
	fig, axes = plt.subplots(2, 2, figsize=(15, 12))
	fig.suptitle('Training Phase Results Visualization', fontsize=16, fontweight='bold')
	
	ax1 = axes[0, 0]
	ax1.hist(train_distances, bins=30, alpha=0.7, color='blue', edgecolor='black')
	ax1.axvline(detector.threshold_strict, color='orange', linestyle='--', linewidth=1.2,
	            label=f'Strict Threshold = {detector.threshold_strict:.4f}')
	ax1.axvline(detector.threshold_loose, color='red', linestyle='--', linewidth=1.2,
	            label=f'Loose Threshold = {detector.threshold_loose:.4f}')
	ax1.axvline(np.mean(train_distances), color='green', linestyle='--', linewidth=1.2,
	            label=f'Mean = {np.mean(train_distances):.4f}')
	ax1.set_xlabel('Anomaly Distance', fontweight='bold')
	ax1.set_ylabel('Frequency', fontweight='bold')
	ax1.set_title('Training Set Distance Distribution', fontweight='bold')
	ax1.legend()
	ax1.grid(True, alpha=0.3)
	
	ax2 = axes[0, 1]
	x_pos = np.arange(len(file_names))
	colors = ['red' if dist > detector.threshold_loose else 'orange' if dist > detector.threshold_strict else 'blue' for
	          dist in train_distances]
	ax2.bar(x_pos, train_distances, color=colors, alpha=0.7)
	ax2.axhline(detector.threshold_strict, color='orange', linestyle='--', linewidth=1.2, label='Strict Threshold')
	ax2.axhline(detector.threshold_loose, color='red', linestyle='--', linewidth=1.2, label='Loose Threshold')
	ax2.set_xlabel('File Index', fontweight='bold')
	ax2.set_ylabel('Anomaly Distance', fontweight='bold')
	ax2.set_title('Anomaly Distance per File', fontweight='bold')
	ax2.legend()
	ax2.grid(True, alpha=0.3)
	
	ax3 = axes[1, 0]
	main_indicators = ['R_eig', 'R_lyap', 'R_svd', 'R_corr']
	weights_list = [detector.weights[name] for name in main_indicators]
	ax3.bar(main_indicators, weights_list, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
	ax3.set_ylabel('Weight Value', fontweight='bold')
	ax3.set_title('Optimal Weights for Four Main Indicators', fontweight='bold')
	ax3.grid(True, alpha=0.3)
	
	ax4 = axes[1, 1]
	means = [detector.normal_profile[name]['mean'] for name in main_indicators]
	stds = [detector.normal_profile[name]['std'] for name in main_indicators]
	ax4.errorbar(np.arange(len(main_indicators)), means, yerr=stds, fmt='o', capsize=5, capthick=2, markersize=8)
	ax4.set_xticks(np.arange(len(main_indicators)))
	ax4.set_xticklabels(main_indicators, fontweight='bold')
	ax4.set_ylabel('Value', fontweight='bold')
	ax4.set_title('Normal Distribution Parameters (Mean ± Std)', fontweight='bold')
	ax4.grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, 'training_results.png'), dpi=600, bbox_inches='tight')
	plt.close()


def visualize_test_results(test_results, output_dir):
	fig, axes = plt.subplots(2, 2, figsize=(15, 12))
	fig.suptitle('Test Phase Results Visualization', fontsize=16, fontweight='bold')
	
	file_names = [r['file_name'] for r in test_results]
	total_distances = [r['total_distance'] for r in test_results]
	is_anomaly = [r['is_anomaly'] for r in test_results]
	anomaly_levels = [r['anomaly_level'] for r in test_results]
	threshold_strict = test_results[0]['threshold_strict']
	threshold_loose = test_results[0]['threshold_loose']
	
	ax1 = axes[0, 0]
	ax1.hist(total_distances, bins=20, alpha=0.7, color='gray', edgecolor='black')
	ax1.axvline(threshold_strict, color='orange', linestyle='--', linewidth=1.2,
	            label=f'Strict Threshold = {threshold_strict:.4f}')
	ax1.axvline(threshold_loose, color='red', linestyle='--', linewidth=1.2,
	            label=f'Loose Threshold = {threshold_loose:.4f}')
	ax1.set_xlabel('Anomaly Distance', fontweight='bold')
	ax1.set_ylabel('Frequency', fontweight='bold')
	ax1.set_title('Test Set Distance Distribution', fontweight='bold')
	ax1.legend()
	ax1.grid(True, alpha=0.3)
	
	ax2 = axes[0, 1]
	
	def get_color(level):
		return 'blue' if level == 'Normal' else 'orange' if level == 'Border/Suspicious' else 'red'
	
	colors_bar = [get_color(level) for level in anomaly_levels]
	bars = ax2.bar(np.arange(len(file_names)), total_distances, color=colors_bar, alpha=0.7)
	ax2.axhline(threshold_strict, color='orange', linestyle='--', linewidth=1.2, label='Strict Threshold')
	ax2.axhline(threshold_loose, color='red', linestyle='--', linewidth=1.2, label='Loose Threshold')
	ax2.set_xlabel('File Index', fontweight='bold')
	ax2.set_ylabel('Anomaly Distance', fontweight='bold')
	ax2.set_title('Anomaly Distance per File (Blue=Normal, Orange=Border, Red=Anomaly)', fontweight='bold')
	ax2.legend()
	ax2.grid(True, alpha=0.3)
	
	ax3 = axes[1, 0]
	main_indicators = ['R_eig', 'R_lyap', 'R_svd', 'R_corr']
	score_matrix = np.array([[r['scores'][name] for name in main_indicators] for r in test_results])
	im = ax3.imshow(score_matrix.T, cmap='RdYlGn_r', aspect='auto')
	ax3.set_yticks(range(len(main_indicators)))
	ax3.set_yticklabels(main_indicators, fontweight='bold')
	ax3.set_xticks(range(len(file_names)))
	ax3.set_xticklabels([f'{i + 1}' for i in range(len(file_names))])
	ax3.set_xlabel('File Index', fontweight='bold')
	ax3.set_title('Indicator Distance Score Heatmap (Green=Normal, Red=Anomaly)', fontweight='bold')
	plt.colorbar(im, ax=ax3)
	
	ax4 = axes[1, 1]
	level_counts = {}
	for level in anomaly_levels: level_counts[level] = level_counts.get(level, 0) + 1
	labels_pie = [l for l in ['Normal', 'Border/Suspicious', 'Anomaly'] if l in level_counts]
	sizes_pie = [level_counts[l] for l in labels_pie]
	colors_pie = [get_color(l) for l in labels_pie]
	if sizes_pie: ax4.pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%', colors=colors_pie, startangle=90,
	                      textprops={'fontweight': 'bold'})
	ax4.set_title(f'Anomaly Detection Results (Total={len(test_results)})', fontweight='bold')
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, 'test_results.png'), dpi=600, bbox_inches='tight')
	plt.close()


def visualize_detailed_test_result(result, output_dir):
	fig, axes = plt.subplots(2, 2, figsize=(15, 12))
	fig.suptitle(f"File: {result['file_name']} - Detailed Analysis", fontsize=16, fontweight='bold')
	
	ax1 = axes[0, 0]
	main_indicators = ['R_eig', 'R_lyap', 'R_svd', 'R_corr']
	indicator_values = [result['indicators'][name] for name in main_indicators]
	ax1.bar(main_indicators, indicator_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], alpha=0.7)
	ax1.set_ylabel('Indicator Value', fontweight='bold')
	ax1.set_title('Raw Values of Four Anomaly Indicators', fontweight='bold')
	ax1.grid(True, alpha=0.3)
	
	ax2 = axes[0, 1]
	score_values = [result['scores'][name] for name in main_indicators]
	colors = ['red' if s > 3.0 else 'orange' if s > 2.0 else 'yellow' if s > 1.0 else 'blue' for s in score_values]
	ax2.bar(main_indicators, score_values, color=colors, alpha=0.7)
	ax2.axhline(3.0, color='red', linestyle='--', linewidth=1.2, label='Severe Anomaly (3σ)')
	ax2.axhline(2.0, color='orange', linestyle='--', linewidth=1.2, label='Anomaly (2σ)')
	ax2.axhline(1.0, color='yellow', linestyle='--', linewidth=1.2, label='Slight Deviation (1σ)')
	ax2.set_ylabel('Distance Score (σ Multiples)', fontweight='bold')
	ax2.set_title('Distance Scores of Four Indicators', fontweight='bold')
	ax2.legend(fontsize=10)
	ax2.grid(True, alpha=0.3)
	
	ax3 = axes[1, 0]
	bars = ax3.bar(['Total Distance', 'Strict Threshold', 'Loose Threshold'],
	               [result['total_distance'], result['threshold_strict'], result['threshold_loose']],
	               color=['red' if result['anomaly_level'] == 'Anomaly' else 'orange' if result[
		                                                                                     'anomaly_level'] == 'Border/Suspicious' else 'blue',
	                      'orange', 'red'], alpha=0.7)
	ax3.set_ylabel('Distance', fontweight='bold')
	ax3.set_title(f"Total Distance vs Thresholds ({result['anomaly_level']})", fontweight='bold')
	ax3.grid(True, alpha=0.3)
	for bar in bars: ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height(), f'{bar.get_height():.4f}',
	                          ha='center', va='bottom', fontsize=10, fontweight='bold')
	
	ax4 = axes[1, 1]
	stability_values = [result['indicators']['stability_slow'], result['indicators']['stability_fast']]
	ax4.bar(['Slow State Stability', 'Fast State Stability'], stability_values, color=['blue', 'red'], alpha=0.7)
	ax4.axhline(1.0, color='green', linestyle='--', linewidth=1.2, label='Ideal Stable Value (1.0)')
	ax4.set_ylabel('Max Eigenvalue Modulus', fontweight='bold')
	ax4.set_title('Fast/Slow State Stability Analysis', fontweight='bold')
	ax4.legend()
	ax4.grid(True, alpha=0.3)
	
	plt.tight_layout()
	safe_filename = result['file_name'].replace('/', '_').replace('\\', '_').replace('.csv', '')
	plt.savefig(os.path.join(output_dir, f'{safe_filename}_detailed.png'), dpi=600, bbox_inches='tight')
	plt.close()


def visualize_koopman_analysis(K, lyap, eigvals, modes, state_type, base_name, output_dir):
	if K is None or lyap is None or eigvals is None or modes is None: return
	fig, axes = plt.subplots(2, 2, figsize=(15, 12))
	fig.suptitle(f'{base_name} - {state_type} Koopman Operator Analysis', fontsize=16, fontweight='bold')
	
	im1 = axes[0, 0].imshow(np.abs(K), cmap='viridis', aspect='auto')
	axes[0, 0].set_title(f'{state_type} Koopman Operator (Absolute Value)', fontweight='bold')
	axes[0, 0].set_xlabel('State Dimension', fontweight='bold')
	axes[0, 0].set_ylabel('State Dimension', fontweight='bold')
	plt.colorbar(im1, ax=axes[0, 0])
	
	axes[0, 1].scatter(eigvals.real, eigvals.imag, c=np.abs(eigvals), cmap='plasma', s=60, alpha=0.8)
	axes[0, 1].add_patch(plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--', linewidth=1.2))
	axes[0, 1].set_title(f'{state_type} Eigenvalue Distribution (Unit Circle)', fontweight='bold')
	axes[0, 1].set_xlabel('Real Part', fontweight='bold')
	axes[0, 1].set_ylabel('Imaginary Part', fontweight='bold')
	axes[0, 1].axhline(0, color='black', linewidth=1)
	axes[0, 1].axvline(0, color='black', linewidth=1)
	axes[0, 1].grid(True, alpha=0.3)
	axes[0, 1].set_aspect('equal')
	
	sorted_lyap = np.sort(lyap.real)[::-1]
	axes[1, 0].bar(range(len(sorted_lyap)), sorted_lyap, color=['red' if x > 0 else 'blue' for x in sorted_lyap],
	               alpha=0.8)
	axes[1, 0].axhline(0, color='black', linewidth=1.2)
	axes[1, 0].set_title(f'{state_type} Lyapunov Exponent Spectrum', fontweight='bold')
	axes[1, 0].set_xlabel('Mode Index', fontweight='bold')
	axes[1, 0].set_ylabel('Lyapunov Exponent', fontweight='bold')
	axes[1, 0].grid(True, alpha=0.3)
	
	mode_energy = np.sum(np.abs(modes) ** 2, axis=0)
	mode_energy_sorted = np.sort(mode_energy)[::-1]
	axes[1, 1].bar(range(len(mode_energy_sorted)), mode_energy_sorted, color='green', alpha=0.8)
	axes[1, 1].set_title(f'{state_type} DMD Mode Energy Distribution', fontweight='bold')
	axes[1, 1].set_xlabel('Mode Index', fontweight='bold')
	axes[1, 1].set_ylabel('Energy', fontweight='bold')
	axes[1, 1].set_yscale('log')
	axes[1, 1].grid(True, alpha=0.3)
	
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f'{base_name}_{state_type}_koopman.png'), dpi=600, bbox_inches='tight')
	plt.close()


def visualize_dmd_modes(modes, state_type, base_name, output_dir):
	if modes is None: return
	n_modes = min(6, modes.shape[1])
	fig, axes = plt.subplots(2, 3, figsize=(18, 10))
	fig.suptitle(f'{base_name} - {state_type} First {n_modes} DMD Modes', fontsize=16, fontweight='bold')
	axes = axes.flatten()
	for i in range(n_modes):
		axes[i].plot(modes[:, i].real, label='Real Part', linewidth=1.2, alpha=0.8)
		axes[i].plot(modes[:, i].imag, label='Imaginary Part', linewidth=1.2, alpha=0.8)
		axes[i].set_title(f'Mode {i + 1}', fontweight='bold')
		axes[i].set_xlabel('State Dimension', fontweight='bold')
		axes[i].set_ylabel('Magnitude', fontweight='bold')
		axes[i].legend()
		axes[i].grid(True, alpha=0.3)
	for i in range(n_modes, 6): axes[i].axis('off')
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f'{base_name}_{state_type}_dmd_modes.png'), dpi=600, bbox_inches='tight')
	plt.close()


def visualize_koopman_prediction(actual, predicted, state_type, base_name, output_dir):
	if actual is None or predicted is None: return
	n_dims = min(3, actual.shape[1])
	fig, axes = plt.subplots(n_dims, 1, figsize=(15, 4 * n_dims))
	if n_dims == 1: axes = [axes]
	fig.suptitle(f'{base_name} - {state_type} Koopman Prediction vs Actual', fontsize=16, fontweight='bold')
	time_steps = np.arange(actual.shape[0])
	for i in range(n_dims):
		axes[i].plot(time_steps, actual[:, i], label='Actual', linewidth=1.2, alpha=0.8, color='black')
		axes[i].plot(time_steps, predicted[:, i], label='Predicted', linewidth=1.2, alpha=0.8, linestyle='--',
		             color='red')
		axes[i].set_title(f'Feature Dimension {i + 1}', fontweight='bold')
		axes[i].set_xlabel('Time Step', fontweight='bold')
		axes[i].set_ylabel('Magnitude', fontweight='bold')
		axes[i].legend()
		axes[i].grid(True, alpha=0.3)
		rmse = np.sqrt(np.mean((actual[:, i] - predicted[:, i]) ** 2))
		axes[i].text(0.02, 0.98, f'RMSE: {rmse:.4f}', transform=axes[i].transAxes, verticalalignment='top', fontsize=12,
		             fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f'{base_name}_{state_type}_prediction.png'), dpi=600, bbox_inches='tight')
	plt.close()


def visualize_comprehensive_analysis(result, file_name, output_dir, fs=1.0):
	from scipy.signal import welch
	from matplotlib import ticker
	
	base_name = file_name.replace('.csv', '')
	
	detail_dir = os.path.join(output_dir, f'{base_name}_details')
	os.makedirs(detail_dir, exist_ok=True)
	
	scaled_data = result['scaled_data']
	slow_states, fast_states = result['slow_states'], result['fast_states']
	slow_pca, fast_pca = result['slow_pca'], result['fast_pca']
	eigenvalues = result['eigenvalues']
	slow_ratio, fast_ratio = result['slow_ratio'], result['fast_ratio']
	variable_contributions = result['variable_contributions']
	slow_indices, fast_indices = result['slow_indices'], result['fast_indices']
	time = np.arange(scaled_data.shape[0])
	
	def get_freq_mag_psd(data, fs):
		n = data.shape[0]
		fft_vals = np.fft.rfft(data, axis=0)
		mag = np.mean(np.abs(fft_vals), axis=1)
		f, psd = welch(data[:, 0], fs, nperseg=min(256, n))
		return np.fft.rfftfreq(n, 1 / fs), mag, f, psd
	
	def get_fft_mag_phase(data, fs):
		n = data.shape[0]
		windowed_data = data * np.hanning(n)[:, np.newaxis]
		fft_vals = np.fft.rfft(windowed_data, axis=0)
		mag = np.mean(np.abs(fft_vals), axis=1)
		phase = np.mean(np.angle(fft_vals), axis=1)
		return np.fft.rfftfreq(n, 1 / fs), mag, phase
	
	freq_orig, mag_orig, f_orig_psd, psd_orig = get_freq_mag_psd(scaled_data, fs)
	fft_freq_orig, fft_mag_orig, fft_phase_orig = get_fft_mag_phase(scaled_data, fs)
	
	freq_slow, mag_slow, f_slow_psd, psd_slow = (None,) * 4
	fft_freq_slow, fft_mag_slow, fft_phase_slow = (None,) * 3
	if len(slow_indices) > 0:
		freq_slow, mag_slow, f_slow_psd, psd_slow = get_freq_mag_psd(slow_states, fs)
		fft_freq_slow, fft_mag_slow, fft_phase_slow = get_fft_mag_phase(slow_states, fs)
	
	freq_fast, mag_fast, f_fast_psd, psd_fast = (None,) * 4
	fft_freq_fast, fft_mag_fast, fft_phase_fast = (None,) * 3
	if len(fast_indices) > 0:
		freq_fast, mag_fast, f_fast_psd, psd_fast = get_freq_mag_psd(fast_states, fs)
		fft_freq_fast, fft_mag_fast, fft_phase_fast = get_fft_mag_phase(fast_states, fs)
	
	@ticker.FuncFormatter
	def log_formatter(x, pos):
		exponent = int(np.floor(np.log10(abs(x)))) if x != 0 else 0
		return f"$10^{{{exponent}}}$"
	
	FIG_SIZE = (8, 6)
	DPI = 600
	
	plt.figure(figsize=FIG_SIZE)
	colors = ['blue' if i in slow_indices else 'red' for i in range(len(eigenvalues))]
	plt.bar(range(len(eigenvalues)), eigenvalues, color=colors, alpha=0.8)
	plt.axhline(y=np.median(eigenvalues), color='black', linestyle='--', linewidth=1.2, label='Median Threshold')
	plt.xlabel('Mode Index', fontweight='bold')
	plt.ylabel('Eigenvalue Magnitude (Singular Value²)', fontweight='bold')
	plt.title('Eigenvalue Spectrum (Slow/Fast State Separation)', fontweight='bold')
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(detail_dir, '01_Eigenvalues_Spectrum.png'), dpi=DPI)
	plt.close()
	
	plt.figure(figsize=FIG_SIZE)
	x = np.arange(len(slow_ratio))
	width = 0.35
	plt.bar(x - width / 2, slow_ratio, width, label='Slow State Contribution', color='blue', alpha=0.8)
	plt.bar(x + width / 2, fast_ratio, width, label='Fast State Contribution', color='red', alpha=0.8)
	plt.xlabel('Variable Index', fontweight='bold')
	plt.ylabel('Contribution Ratio (Normalized)', fontweight='bold')
	plt.title('Slow/Fast State Contribution Ratio of Variables', fontweight='bold')
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(detail_dir, '02_Contribution_Ratio.png'), dpi=DPI)
	plt.close()
	
	plt.figure(figsize=FIG_SIZE)
	im = plt.imshow(variable_contributions, aspect='auto', cmap='viridis', vmin=0)
	plt.xlabel('Mode Index', fontweight='bold')
	plt.ylabel('Variable Index', fontweight='bold')
	plt.title('Variable Contribution in Dynamic Modes', fontweight='bold')
	cb3 = plt.colorbar(im, label='Contribution (Absolute Value)')
	cb3.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
	plt.tight_layout()
	plt.savefig(os.path.join(detail_dir, '03_Contribution_Heatmap.png'), dpi=DPI)
	plt.close()
	
	plt.figure(figsize=(10, 5))
	if len(slow_indices) > 0:
		for i in range(min(3, slow_states.shape[1])):
			plt.plot(time[:-1], slow_states[:, i], label=f'Slow Mode {i + 1}', linewidth=1.2, alpha=0.8)
	else:
		plt.text(0.5, 0.5, 'No Slow State Modes Detected', ha='center', va='center', fontsize=14, fontweight='bold')
	plt.xlabel('Time Step', fontweight='bold')
	plt.ylabel('Amplitude (Normalized)', fontweight='bold')
	plt.title('Slow State Time Evolution (Top 3 Modes)', fontweight='bold')
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(detail_dir, '04_Slow_State_Time_Evolution.png'), dpi=DPI)
	plt.close()
	
	plt.figure(figsize=(10, 5))
	if len(fast_indices) > 0:
		for i in range(min(3, fast_states.shape[1])):
			plt.plot(time[:-1], fast_states[:, i], label=f'Fast Mode {i + 1}', linewidth=1.2, alpha=0.8)
	else:
		plt.text(0.5, 0.5, 'No Fast State Modes Detected', ha='center', va='center', fontsize=14, fontweight='bold')
	plt.xlabel('Time Step', fontweight='bold')
	plt.ylabel('Amplitude (Normalized)', fontweight='bold')
	plt.title('Fast State Time Evolution (Top 3 Modes)', fontweight='bold')
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(detail_dir, '05_Fast_State_Time_Evolution.png'), dpi=DPI)
	plt.close()
	
	plt.figure(figsize=(7, 6))
	if len(slow_indices) > 0:
		scatter = plt.scatter(slow_pca[:, 0], slow_pca[:, 1], c=time[:-1], cmap='viridis', alpha=0.8, s=40)
		cb6 = plt.colorbar(scatter, label='Time Step')
		cb6.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
	else:
		plt.text(0.5, 0.5, 'No Slow State Phase Space Data', ha='center', va='center')
	plt.xlabel('PCA Component 1', fontweight='bold')
	plt.ylabel('PCA Component 2', fontweight='bold')
	plt.title('Slow State Phase Space (PCA Reduction)', fontweight='bold')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(detail_dir, '06_Slow_State_Phase_Space.png'), dpi=DPI)
	plt.close()
	
	plt.figure(figsize=(7, 6))
	if len(fast_indices) > 0:
		scatter = plt.scatter(fast_pca[:, 0], fast_pca[:, 1], c=time[:-1], cmap='plasma', alpha=0.8, s=40)
		cb7 = plt.colorbar(scatter, label='Time Step')
		cb7.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
	else:
		plt.text(0.5, 0.5, 'No Fast State Phase Space Data', ha='center', va='center')
	plt.xlabel('PCA Component 1', fontweight='bold')
	plt.ylabel('PCA Component 2', fontweight='bold')
	plt.title('Fast State Phase Space (PCA Reduction)', fontweight='bold')
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(detail_dir, '07_Fast_State_Phase_Space.png'), dpi=DPI)
	plt.close()
	
	plt.figure(figsize=FIG_SIZE)
	plt.plot(freq_orig, mag_orig, label='Original Signal', color='black', alpha=0.8, linewidth=1.2)
	if freq_slow is not None:
		plt.plot(freq_slow, mag_slow, label='Slow State', color='blue', alpha=0.8, linewidth=1.2)
	if freq_fast is not None:
		plt.plot(freq_fast, mag_fast, label='Fast State', color='red', alpha=0.8, linewidth=1.2)
	plt.xlabel('Frequency [Hz]', fontweight='bold')
	plt.ylabel('Magnitude', fontweight='bold')
	plt.title('Frequency Domain Analysis - Magnitude Spectrum', fontweight='bold')
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(detail_dir, '08_Frequency_Magnitude.png'), dpi=DPI)
	plt.close()
	
	plt.figure(figsize=FIG_SIZE)
	plt.plot(fft_freq_orig, fft_mag_orig, label='Original Signal', color='black', alpha=0.8, linewidth=1.2)
	if fft_freq_slow is not None:
		plt.plot(fft_freq_slow, fft_mag_slow, label='Slow State', color='blue', alpha=0.8, linewidth=1.2)
	if fft_freq_fast is not None:
		plt.plot(fft_freq_fast, fft_mag_fast, label='Fast State', color='red', alpha=0.8, linewidth=1.2)
	plt.xlabel('Frequency [Hz]', fontweight='bold')
	plt.ylabel('Magnitude (Hanning Window)', fontweight='bold')
	plt.title('FFT Analysis - Magnitude Spectrum (Hanning Window)', fontweight='bold')
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(detail_dir, '09_FFT_Magnitude.png'), dpi=DPI)
	plt.close()
	
	plt.figure(figsize=FIG_SIZE)
	plt.plot(fft_freq_orig, fft_phase_orig, label='Original Signal', color='black', alpha=0.8, linewidth=1.2)
	if fft_freq_slow is not None:
		plt.plot(fft_freq_slow, fft_phase_slow, label='Slow State', color='blue', alpha=0.8, linewidth=1.2)
	if fft_freq_fast is not None:
		plt.plot(fft_freq_fast, fft_phase_fast, label='Fast State', color='red', alpha=0.8, linewidth=1.2)
	plt.xlabel('Frequency [Hz]', fontweight='bold')
	plt.ylabel('Phase [rad]', fontweight='bold')
	plt.title('FFT Analysis - Phase Spectrum', fontweight='bold')
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(detail_dir, '10_FFT_Phase.png'), dpi=DPI)
	plt.close()
	
	plt.figure(figsize=(10, 6))
	if len(slow_indices) > 0 and len(fast_indices) > 0:
		plt.loglog(f_orig_psd, psd_orig, label='Original Signal', color='black', alpha=0.8, linewidth=1.20)
		plt.loglog(f_slow_psd, psd_slow, label='Slow State', color='blue', alpha=0.8, linewidth=1.20)
		plt.loglog(f_fast_psd, psd_fast, label='Fast State', color='red', alpha=0.8, linewidth=1.2)
		plt.gca().xaxis.set_major_formatter(log_formatter)
		plt.gca().yaxis.set_major_formatter(log_formatter)
	else:
		plt.text(0.5, 0.5, 'Insufficient Data for PSD Calculation', ha='center', va='center', fontsize=14)
	plt.xlabel('Frequency [Hz]', fontweight='bold')
	plt.ylabel('Power Spectral Density PSD [V²/Hz]', fontweight='bold')
	plt.title('Power Spectral Density (Welch Method)', fontweight='bold')
	plt.legend()
	plt.grid(True, which="both", ls="--", alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(detail_dir, '11_PSD.png'), dpi=DPI)
	plt.close()
	
	plt.figure(figsize=(10, 6))
	if len(slow_indices) > 0 and len(fast_indices) > 0:
		plt.loglog(f_orig_psd, psd_orig, label='Original Signal', color='black', alpha=0.8, linewidth=1.2)
		plt.loglog(f_slow_psd, psd_slow, label='Slow State', color='blue', alpha=0.8, linewidth=1.2)
		plt.loglog(f_fast_psd, psd_fast, label='Fast State', color='red', alpha=0.8, linewidth=1.2)
		plt.gca().xaxis.set_major_formatter(log_formatter)
		plt.gca().yaxis.set_major_formatter(log_formatter)
	else:
		plt.text(0.5, 0.5, 'Insufficient Data for PSD Comparison', ha='center', va='center', fontsize=14)
	plt.xlabel('Frequency [Hz]', fontweight='bold')
	plt.ylabel('Power Spectral Density PSD [V²/Hz]', fontweight='bold')
	plt.title('Comprehensive Power Spectral Density Comparison', fontweight='bold')
	plt.legend()
	plt.grid(True, which="both", ls="--", alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(detail_dir, '12_PSD_Comparison.png'), dpi=DPI)
	plt.close()
	
	visualize_koopman_analysis(result['K_slow'], result['lyap_slow'], result['eigvals_slow'], result['modes_slow'],
	                           "Slow", base_name, detail_dir)
	visualize_koopman_analysis(result['K_fast'], result['lyap_fast'], result['eigvals_fast'], result['modes_fast'],
	                           "Fast", base_name, detail_dir)
	visualize_dmd_modes(result['modes_slow'], "Slow", base_name, detail_dir)
	visualize_dmd_modes(result['modes_fast'], "Fast", base_name, detail_dir)
	
	if result.get('predicted_slow') is not None:
		visualize_koopman_prediction(slow_states[1:, :], result['predicted_slow'], "Slow", base_name, detail_dir)
	if result.get('predicted_fast') is not None:
		visualize_koopman_prediction(fast_states[1:, :], result['predicted_fast'], "Fast", base_name, detail_dir)
	
	print(f"✅ All high-resolution detailed charts (600 DPI) for {base_name} saved to: {detail_dir}")


def load_and_process_data(directory_path):
	csv_files = sorted(glob.glob(os.path.join(directory_path, "*.csv")))
	
	if not csv_files:
		raise ValueError("No CSV files found in the specified directory")
	
	all_data = []
	file_names = []
	for file in csv_files:
		df = pd.read_csv(file)
		data = df.iloc[:, :12].values
		column_names = df.columns[:12].tolist()
		
		all_data.append(data)
		file_names.append(os.path.basename(file))
	
	processed_data = []
	for data in all_data:
		processed_data.append(np.nan_to_num(data))
	
	return processed_data, file_names


def compute_koopman_operator_dmd(X, Y, rank=None):

	U, s, Vh = svd(X, full_matrices=False)
	rank = len(s)
	U_r, s_r, Vh_r = U[:, :rank], s[:rank], Vh[:rank, :]
	
	A_tilde = U_r.T @ Y @ Vh_r.T @ np.diag(1.0 / s_r)
	eigenvalues, eigenvectors = eig(A_tilde)
	modes = Y @ Vh_r.T @ np.diag(1.0 / s_r) @ eigenvectors
	
	valid_indices = []
	residual_threshold = 2.0
	
	V_Sigma_inv = Vh_r.T @ np.diag(1.0 / s_r)
	for i in range(len(eigenvalues)):
		lam = eigenvalues[i]
		w = eigenvectors[:, i]
		term1 = Y @ V_Sigma_inv @ w
		term2 = lam * (U_r @ w)
		residual = np.linalg.norm(term1 - term2) / (np.linalg.norm(term2) + 1e-10)

		if residual < residual_threshold:
			valid_indices.append(i)
	if len(valid_indices) == 0:
		valid_indices = [np.argmin(np.abs(eigenvalues - 1.0))]
	
	filtered_eigenvalues = eigenvalues[valid_indices]
	filtered_modes = modes[:, valid_indices]

	K_reduced = A_tilde
	predicted = Y
	return K_reduced, filtered_modes, filtered_eigenvalues, predicted

def compute_anomaly_indicators(result):
	indicators = {}
	
	if result['eigvals_slow'] is not None and result['eigvals_fast'] is not None:
		eig_slow_geom = np.exp(np.mean(np.log(np.abs(result['eigvals_slow']) + 1e-10)))
		eig_fast_geom = np.exp(np.mean(np.log(np.abs(result['eigvals_fast']) + 1e-10)))
		indicators['R_eig'] = eig_fast_geom / (eig_slow_geom + 1e-10)
		
		indicators['stability_slow'] = np.max(np.abs(result['eigvals_slow']))
		indicators['stability_fast'] = np.max(np.abs(result['eigvals_fast']))
	else:
		indicators['R_eig'] = 0.0
		indicators['stability_slow'] = 0.0
		indicators['stability_fast'] = 0.0
	
	if result['lyap_slow'] is not None and result['lyap_fast'] is not None:
		lyap_slow_max = np.max(np.abs(result['lyap_slow']))
		lyap_fast_max = np.max(np.abs(result['lyap_fast']))
		indicators['R_lyap'] = lyap_fast_max / (lyap_slow_max + 1e-10)
	else:
		indicators['R_lyap'] = 0.0
	
	if result['total_energy'] > 0:
		indicators['R_svd'] = result['fast_energy'] / (result['slow_energy'] + 1e-10)
	else:
		indicators['R_svd'] = 0.0
	
	if result['corr_slow_mean'] > 0 and result['corr_fast_mean'] > 0:
		indicators['R_corr'] = result['corr_fast_mean'] / (result['corr_slow_mean'] + 1e-10)
	else:
		indicators['R_corr'] = 0.0
	
	return indicators


def compute_distance_score(value, mean, std):
	return abs((value - mean) / (std + 1e-10))


class AnomalyDetector:
	def __init__(self):
		self.normal_profile = None
		self.weights = None
		self.threshold_strict = None
		self.threshold_loose = None
		self.threshold_margin = 0.1
	
	def train(self, data_list, file_names, lambda_reg=1.5, search_space=None,
	          threshold_percentile=97, use_robust_threshold=True):
		print("=" * 60)
		print("Start training...")
		if search_space is None: search_space = np.linspace(0.3, 4.0, 15)
		
		all_indicators = []
		for i, (data, fname) in enumerate(zip(data_list, file_names)):
			result = separate_fast_slow_states(data, dmd_rank=10, is_online=False)
			all_indicators.append(compute_anomaly_indicators(result))
		
		indicator_names = ['R_eig', 'R_lyap', 'R_svd', 'R_corr', 'stability_slow', 'stability_fast']
		self.normal_profile = {}
		for name in indicator_names:
			values = np.array([ind.get(name, 0) for ind in all_indicators])
			values = values[np.isfinite(values)]
			q25, q75 = np.percentile(values, 25), np.percentile(values, 75)
			self.normal_profile[name] = {'mean': np.mean(values), 'std': np.std(values) + 1e-10,
			                             'median': np.median(values), 'q25': q25, 'q75': q75, 'iqr': q75 - q25}
		
		distance_scores = [{name: compute_distance_score(ind.get(name, 0), self.normal_profile[name]['mean'],
		                                                 self.normal_profile[name]['std']) for name in indicator_names}
		                   for ind in all_indicators]
		
		best_loss, best_weights = float('inf'), None
		main_indicators = ['R_eig', 'R_lyap', 'R_svd', 'R_corr']
		
		for w1, w2, w3, w4 in product(search_space, repeat=4):
			weights = {'R_eig': w1, 'R_lyap': w2, 'R_svd': w3, 'R_corr': w4, 'stability_slow': 0.8,
			           'stability_fast': 0.8}
			weight_sum = sum(weights[n] for n in main_indicators)
			
			total_distances = [sum(weights[n] * s[n] for n in main_indicators) / weight_sum + abs(
				s['stability_slow'] - 1.0) * 0.5 + abs(s['stability_fast'] - 1.0) * 0.5 for s in distance_scores]
			mean_dist = np.mean(total_distances)
			loss = mean_dist + lambda_reg * np.std(total_distances) - max(0, 1.0 - mean_dist) * 2.0
			if loss < best_loss: best_loss, best_weights = loss, weights.copy()
		
		self.weights = best_weights
		final_distances = [
			sum(self.weights[n] * s[n] for n in main_indicators) / sum(self.weights[n] for n in main_indicators) + abs(
				s['stability_slow'] - 1.0) * 0.5 + abs(s['stability_fast'] - 1.0) * 0.5 for s in distance_scores]
		
		if use_robust_threshold:
			q25, q75 = np.percentile(final_distances, 25), np.percentile(final_distances, 75)
			self.threshold_strict, self.threshold_loose = q75 + 1.5 * (q75 - q25), q75 + 3.0 * (q75 - q25)
		
		return final_distances
	
	def predict(self, data, return_details=False):
		
		result = separate_fast_slow_states(data, dmd_rank=10, is_online=True)
		indicators = compute_anomaly_indicators(result)
		indicator_names = ['R_eig', 'R_lyap', 'R_svd', 'R_corr', 'stability_slow', 'stability_fast']
		
		scores = {}
		for name in indicator_names:
			if name in indicators:
				raw = compute_distance_score(indicators[name], self.normal_profile.get(name, {}).get('mean', 0),
				                             self.normal_profile.get(name, {}).get('std', 1))
				scores[name] = raw if raw <= 2.0 else raw * 1.5
		
		main_inds = ['R_eig', 'R_lyap', 'R_svd', 'R_corr']
		weight_sum = sum(self.weights.get(n, 1.0) for n in main_inds)
		
		total_distance = sum(self.weights.get(n, 1.0) * scores.get(n, 0) for n in main_inds) / weight_sum
		total_distance += abs(scores.get('stability_slow', 0) - 1.0) * 0.5
		total_distance += abs(scores.get('stability_fast', 0) - 1.0) * 0.5
		
		is_anomaly = total_distance > self.threshold_strict
		
		if total_distance <= self.threshold_strict * 0.8:
			anomaly_level, anomaly_degree = 'Normal', 0.0
		elif total_distance <= self.threshold_strict:
			anomaly_level, anomaly_degree = 'Border/Suspicious', (total_distance - self.threshold_strict * 0.8) / (
					self.threshold_strict * 0.2) * 0.3
		elif total_distance <= self.threshold_loose:
			anomaly_level, anomaly_degree = 'Mild abnormality', 0.3 + (total_distance - self.threshold_strict) / (
					self.threshold_loose - self.threshold_strict) * 0.4
		else:
			anomaly_level, anomaly_degree = 'Significant abnormality', 0.7 + min(0.3, (
					total_distance - self.threshold_loose) / self.threshold_loose * 0.3)
		
		if return_details: return is_anomaly, total_distance, {'indicators': indicators, 'scores': scores,
		                                                       'total_distance': total_distance,
		                                                       'threshold_strict': self.threshold_strict,
		                                                       'threshold_loose': self.threshold_loose,
		                                                       'is_anomaly': is_anomaly, 'anomaly_level': anomaly_level,
		                                                       'anomaly_degree': anomaly_degree}
		return is_anomaly, total_distance
	
	def save_model(self, filepath):
		with open(filepath, 'w') as f: json.dump({'normal_profile': self.normal_profile, 'weights': self.weights,
		                                          'threshold_strict': float(self.threshold_strict),
		                                          'threshold_loose': float(self.threshold_loose),
		                                          'threshold_margin': float(self.threshold_margin)}, f, indent=2)
	
	def load_model(self, filepath):
		with open(filepath, 'r') as f:
			m = json.load(f)
			self.normal_profile, self.weights, self.threshold_strict, self.threshold_loose, self.threshold_margin = m[
				'normal_profile'], m['weights'], m['threshold_strict'], m['threshold_loose'], m.get('threshold_margin',
			                                                                                        0.1)


def compute_anomaly_indicators(result):
	indicators = {}
	
	if result['eigvals_slow'] is not None and result['eigvals_fast'] is not None:
		eig_slow_geom = np.exp(np.mean(np.log(np.abs(result['eigvals_slow']) + 1e-10)))
		eig_fast_geom = np.exp(np.mean(np.log(np.abs(result['eigvals_fast']) + 1e-10)))
		indicators['R_eig'] = eig_fast_geom / (eig_slow_geom + 1e-10)
		
		indicators['stability_slow'] = np.max(np.abs(result['eigvals_slow']))
		indicators['stability_fast'] = np.max(np.abs(result['eigvals_fast']))
	else:
		indicators['R_eig'] = 0.0
		indicators['stability_slow'] = 0.0
		indicators['stability_fast'] = 0.0
	
	if result['lyap_slow'] is not None and result['lyap_fast'] is not None:
		lyap_slow_max = np.max(np.abs(result['lyap_slow']))
		lyap_fast_max = np.max(np.abs(result['lyap_fast']))
		indicators['R_lyap'] = lyap_fast_max / (lyap_slow_max + 1e-10)
	else:
		indicators['R_lyap'] = 0.0
	
	if result['total_energy'] > 0:
		indicators['R_svd'] = result['fast_energy'] / (result['slow_energy'] + 1e-10)
	else:
		indicators['R_svd'] = 0.0
	
	if result['corr_slow_mean'] > 0 and result['corr_fast_mean'] > 0:
		indicators['R_corr'] = result['corr_fast_mean'] / (result['corr_slow_mean'] + 1e-10)
	else:
		indicators['R_corr'] = 0.0
	
	return indicators


def compute_koopman_operator_online(X, Y, rho=0.99):
	d = X.shape[1]
	odmd = OnlineDMD(d=d, rho=rho)
	
	for t in range(X.shape[0]):
		odmd.update(X[t], Y[t])
	
	K = odmd.K
	eigenvalues, modes, lyapunov_exponents = odmd.get_dynamics()
	
	predicted = None
	try:
		predicted = X @ K.T
	except Exception as e:
		pass
	
	return K, modes, eigenvalues, predicted, lyapunov_exponents


def compute_lyapunov_exponents(eigenvalues, dt=1.0):
	lyapunov_exponents = np.log(np.abs(eigenvalues)) / dt
	return lyapunov_exponents


def separate_fast_slow_states(data, epsilon_threshold=None, dmd_rank=None, is_online=False):
	scaler = StandardScaler()
	scaled_data = scaler.fit_transform(data)
	
	diff_data = np.diff(scaled_data, axis=0)
	
	U, s, Vt = svd(diff_data, full_matrices=False)
	
	eigenvalues = s ** 2
	
	if epsilon_threshold is None:
		epsilon_threshold = np.median(eigenvalues)
	
	slow_indices = np.where(eigenvalues < epsilon_threshold)[0]
	fast_indices = np.where(eigenvalues >= epsilon_threshold)[0]
	
	variable_contributions = np.abs(Vt.T)
	
	if len(slow_indices) > 0:
		slow_contributions = np.sum(variable_contributions[:, slow_indices], axis=1)
	else:
		slow_contributions = np.zeros(variable_contributions.shape[0])
	
	if len(fast_indices) > 0:
		fast_contributions = np.sum(variable_contributions[:, fast_indices], axis=1)
	else:
		fast_contributions = np.zeros(variable_contributions.shape[0])
	
	total_contributions = slow_contributions + fast_contributions
	total_contributions[total_contributions == 0] = 1e-10
	
	slow_ratio = slow_contributions / total_contributions
	fast_ratio = fast_contributions / total_contributions
	
	if len(slow_indices) > 0:
		slow_states = U[:, slow_indices] @ np.diag(s[slow_indices]) @ Vt[slow_indices, :]
	else:
		slow_states = np.zeros((U.shape[0], Vt.shape[1]))
	
	if len(fast_indices) > 0:
		fast_states = U[:, fast_indices] @ np.diag(s[fast_indices]) @ Vt[fast_indices, :]
	else:
		fast_states = np.zeros((U.shape[0], Vt.shape[1]))
	
	if len(slow_indices) > 0:
		X_slow = slow_states[:-1, :]
		Y_slow = slow_states[1:, :]
		if is_online:
			K_slow, modes_slow, eigvals_slow, predicted_slow, lyap_slow = compute_koopman_operator_online(X_slow,
			                                                                                              Y_slow)
		else:
			K_slow, modes_slow, eigvals_slow, predicted_slow = compute_koopman_operator_dmd(X_slow, Y_slow, dmd_rank)
			lyap_slow = compute_lyapunov_exponents(eigvals_slow)
	else:
		K_slow, modes_slow, eigvals_slow, lyap_slow, predicted_slow = None, None, None, None, None
	
	if len(fast_indices) > 0:
		X_fast = fast_states[:-1, :]
		Y_fast = fast_states[1:, :]
		if is_online:
			K_fast, modes_fast, eigvals_fast, predicted_fast, lyap_fast = compute_koopman_operator_online(X_fast,
			                                                                                              Y_fast)
		else:
			K_fast, modes_fast, eigvals_fast, predicted_fast = compute_koopman_operator_dmd(X_fast, Y_fast, dmd_rank)
			lyap_fast = compute_lyapunov_exponents(eigvals_fast)
	else:
		K_fast, modes_fast, eigvals_fast, lyap_fast, predicted_fast = None, None, None, None, None
	
	pca = PCA(n_components=2)
	if len(slow_indices) > 0:
		slow_pca = pca.fit_transform(slow_states)
	else:
		slow_pca = np.zeros((slow_states.shape[0], 2))
	
	if len(fast_indices) > 0:
		fast_pca = pca.fit_transform(fast_states)
	else:
		fast_pca = np.zeros((fast_states.shape[0], 2))
	
	total_energy = np.sum(s ** 2)
	slow_energy = np.sum(s[slow_indices] ** 2) if len(slow_indices) > 0 else 0
	fast_energy = np.sum(s[fast_indices] ** 2) if len(fast_indices) > 0 else 0
	
	if len(slow_indices) > 0:
		corr_slow = np.corrcoef(slow_states.T)
		corr_slow_mean = np.mean(np.abs(corr_slow[np.triu_indices_from(corr_slow, k=1)]))
	else:
		corr_slow_mean = 0
	
	if len(fast_indices) > 0:
		corr_fast = np.corrcoef(fast_states.T)
		corr_fast_mean = np.mean(np.abs(corr_fast[np.triu_indices_from(corr_fast, k=1)]))
	else:
		corr_fast_mean = 0
	
	return {
		'slow_states': slow_states,
		'fast_states': fast_states,
		'slow_pca': slow_pca,
		'fast_pca': fast_pca,
		'eigenvalues': eigenvalues,
		'slow_ratio': slow_ratio,
		'fast_ratio': fast_ratio,
		'variable_contributions': variable_contributions,
		'slow_indices': slow_indices,
		'fast_indices': fast_indices,
		'K_slow': K_slow,
		'K_fast': K_fast,
		'lyap_slow': lyap_slow,
		'lyap_fast': lyap_fast,
		'eigvals_slow': eigvals_slow,
		'eigvals_fast': eigvals_fast,
		'modes_slow': modes_slow,
		'modes_fast': modes_fast,
		'scaled_data': scaled_data,
		'slow_energy': slow_energy,
		'fast_energy': fast_energy,
		'total_energy': total_energy,
		'corr_slow_mean': corr_slow_mean,
		'corr_fast_mean': corr_fast_mean,
		'predicted_slow': predicted_slow,
		'predicted_fast': predicted_fast
	}


def compute_anomaly_indicators(result):
	indicators = {}
	
	if result['eigvals_slow'] is not None and result['eigvals_fast'] is not None:
		eig_slow_geom = np.exp(np.mean(np.log(np.abs(result['eigvals_slow']) + 1e-10)))
		eig_fast_geom = np.exp(np.mean(np.log(np.abs(result['eigvals_fast']) + 1e-10)))
		indicators['R_eig'] = eig_fast_geom / (eig_slow_geom + 1e-10)
		
		# 特征值分布偏度指标
		eig_slow_skew = np.mean((result['eigvals_slow'].real - np.mean(result['eigvals_slow'].real)) ** 3)
		eig_fast_skew = np.mean((result['eigvals_fast'].real - np.mean(result['eigvals_fast'].real)) ** 3)
		indicators['eig_skew_ratio'] = abs(eig_fast_skew) / (abs(eig_slow_skew) + 1e-10)
		
		# 稳定性指标
		indicators['stability_slow'] = np.max(np.abs(result['eigvals_slow']))
		indicators['stability_fast'] = np.max(np.abs(result['eigvals_fast']))
	else:
		indicators['R_eig'] = 0.0
		indicators['eig_skew_ratio'] = 0.0
		indicators['stability_slow'] = 0.0
		indicators['stability_fast'] = 0.0
	
	# 2. 李雅普诺夫指数 (使用最大值)
	if result['lyap_slow'] is not None and result['lyap_fast'] is not None:
		lyap_slow_max = np.max(np.abs(result['lyap_slow']))
		lyap_fast_max = np.max(np.abs(result['lyap_fast']))
		indicators['R_lyap'] = lyap_fast_max / (lyap_slow_max + 1e-10)
	else:
		indicators['R_lyap'] = 0.0
	
	# 3. 预测误差指标
	if result.get('predicted_slow') is not None and result.get('predicted_fast') is not None:
		actual_slow = result['slow_states'][1:, :]
		actual_fast = result['fast_states'][1:, :]
		pred_error_slow = np.mean(np.abs(actual_slow - result['predicted_slow']))
		pred_error_fast = np.mean(np.abs(actual_fast - result['predicted_fast']))
		indicators['pred_error_ratio'] = pred_error_fast / (pred_error_slow + 1e-10)
	else:
		indicators['pred_error_ratio'] = 0.0
	
	# 4. 频率域异常指标
	from scipy.signal import welch
	if result['fast_states'].shape[0] > 0:
		f_fast, Pxx_fast = welch(result['fast_states'][:, 0], fs=1.0)
		freq_threshold = 0.1
		high_freq_idx = f_fast > freq_threshold
		if np.sum(high_freq_idx) > 0:
			indicators['high_freq_power'] = np.sum(Pxx_fast[high_freq_idx]) / np.sum(Pxx_fast)
		else:
			indicators['high_freq_power'] = 0.0
	else:
		indicators['high_freq_power'] = 0.0
	
	# 5. SVD能量指标
	if result['total_energy'] > 0:
		indicators['R_svd'] = result['fast_energy'] / (result['slow_energy'] + 1e-10)
	else:
		indicators['R_svd'] = 0.0
	
	# 6. 相关性指标
	if result['corr_slow_mean'] > 0 and result['corr_fast_mean'] > 0:
		indicators['R_corr'] = result['corr_fast_mean'] / (result['corr_slow_mean'] + 1e-10)
	else:
		indicators['R_corr'] = 0.0
	
	return indicators


def compute_frequency_domain_analysis(data, fs=1.0):
	from scipy.signal import welch
	n = data.shape[0]
	freqs = np.fft.rfftfreq(n, 1 / fs)
	
	fft_vals = np.fft.rfft(data, axis=0)
	magnitude = np.mean(np.abs(fft_vals), axis=1)
	
	f, psd = welch(data[:, 0], fs, nperseg=min(256, n))
	
	return freqs, magnitude, psd


def compute_fft_analysis(data, fs=1.0):
	n = data.shape[0]
	
	window = np.hanning(n)[:, np.newaxis]
	windowed_data = data * window
	
	fft_vals = np.fft.rfft(windowed_data, axis=0)
	freqs = np.fft.rfftfreq(n, 1 / fs)
	
	magnitude = np.mean(np.abs(fft_vals), axis=1)
	phase = np.mean(np.angle(fft_vals), axis=1)
	
	return freqs, magnitude, phase


def main_train():
	normal_data_directory = "./train"
	output_directory = "output/CAFUC1"
	
	model_path = os.path.join(output_directory, "anomaly_model.json")
	
	os.makedirs(output_directory, exist_ok=True)
	
	try:
		print("Loading normal data...")
		data_list, file_names = load_and_process_data(normal_data_directory)
		print(f"Found {len(data_list)} normal data files")
		
		detector = AnomalyDetector()
		train_distances = detector.train(
			data_list,
			file_names,
			lambda_reg=1.5,
			threshold_percentile=97,
			use_robust_threshold=True
		)
		
		detector.save_model(model_path)
		visualize_training_results(detector, train_distances, file_names, output_directory)
		
		print("\nTraining phase completed!")
	
	except Exception as e:
		print(f"Error: {e}")
		import traceback
		traceback.print_exc()


def main_test():
	test_data_directory = "./test1"
	output_directory = "output/CAFUC1"
	
	model_path = os.path.join(output_directory, "anomaly_model.json")
	
	os.makedirs(output_directory, exist_ok=True)
	
	try:
		print("Loading trained model...")
		detector = AnomalyDetector()
		detector.load_model(model_path)
		
		print(f"\nModel Thresholds:")
		print(f"  Strict Threshold: {detector.threshold_strict:.4f}")
		print(f"  Loose Threshold: {detector.threshold_loose:.4f}")
		
		print("\nLoading test data...")
		data_list, file_names = load_and_process_data(test_data_directory)
		
		file_paths = sorted(glob.glob(os.path.join(test_data_directory, "*.csv")))
		file_paths_dict = {os.path.basename(fp): fp for fp in file_paths}
		full_file_paths = [file_paths_dict[fn] for fn in file_names]
		
		print(f"Found {len(data_list)} test files")
		
		print("\nStarting anomaly detection...")
		test_results = []
		predictions = []
		anomaly_scores = []
		
		for i, (data, fname) in enumerate(zip(data_list, file_names)):
			print(f"\nTesting file {i + 1}/{len(data_list)}: {fname}")
			is_anomaly, total_distance, details = detector.predict(data, return_details=True)
			
			predictions.append(is_anomaly)
			anomaly_scores.append(total_distance)
			
			print(f"  -> Total Anomaly Score: {total_distance:.4f} ({details['anomaly_level']})")
			
			individual_scores = details['scores']
			scores_str = ", ".join([f"{k}: {v:.2f}" for k, v in individual_scores.items()])
			print(f"  -> Indicator Distance Scores (Z-score): {{ {scores_str} }}")
			
			result = {
				'file_name': fname,
				'is_anomaly': is_anomaly,
				'total_distance': total_distance,
				'threshold_strict': details['threshold_strict'],
				'threshold_loose': details['threshold_loose'],
				'indicators': details['indicators'],
				'scores': details['scores'],
				'anomaly_level': details['anomaly_level'],
				'anomaly_degree': details['anomaly_degree']
			}
			
			test_results.append(result)
			
			fs_result = separate_fast_slow_states(data, dmd_rank=10)
			visualize_comprehensive_analysis(fs_result, fname, output_directory, fs=1.0)
			visualize_detailed_test_result(result, output_directory)
		
		print("\nCalculating model performance metrics...")
		metrics, true_labels = evaluate_accuracy(predictions, full_file_paths, anomaly_scores)
		
		print_evaluation_report(metrics, predictions, true_labels, file_names)
		
		visualize_evaluation_results(metrics, predictions, true_labels, file_names, output_directory, anomaly_scores)
		
		metrics_path = os.path.join(output_directory, 'evaluation_metrics.json')
		with open(metrics_path, 'w', encoding='utf-8') as f:
			json.dump(metrics, f, indent=2, ensure_ascii=False)
		print(f"\nEvaluation metrics saved to: {metrics_path}")
		
		visualize_test_results(test_results, output_directory)
		
		print("\n" + "=" * 40)
		print("           TEST RESULTS SUMMARY")
		print("=" * 40)
		
		summary_df = pd.DataFrame([
			{
				"File Name": r['file_name'],
				"True Label": "Anomaly" if true else "Normal",
				"Total Anomaly Score": r['total_distance'],
				"Result": r['anomaly_level'],
				"Correct": "✓" if (r['is_anomaly'] == true) else "✗"
			}
			for r, true in zip(test_results, true_labels)
		])
		
		summary_df = summary_df.sort_values(by="Total Anomaly Score", ascending=False).reset_index(drop=True)
		print(summary_df.to_string())
		print("=" * 40)
		
		results_df = pd.DataFrame([
			{
				'File Name': r['file_name'],
				'True Label': "Anomaly" if true else "Normal",
				'Anomaly Level': r['anomaly_level'],
				'Total Distance': r['total_distance'],
				'Anomaly Degree(%)': r['anomaly_degree'] * 100,
				'Correct': "✓" if (r['is_anomaly'] == true) else "✗",
				'Strict Threshold': r['threshold_strict'],
				'Loose Threshold': r['threshold_loose'],
				'R_eig Distance': r['scores']['R_eig'],
				'R_lyap Distance': r['scores']['R_lyap'],
				'R_svd Distance': r['scores']['R_svd'],
				'R_corr Distance': r['scores']['R_corr'],
				'Slow State Stability': r['indicators']['stability_slow'],
				'Fast State Stability': r['indicators']['stability_fast']
			}
			for r, true in zip(test_results, true_labels)
		])
		
		csv_path = os.path.join(output_directory, 'test_results.csv')
		results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
		print(f"\nDetailed test results saved to: {csv_path}")
	
	except Exception as e:
		print(f"Error: {e}")
		import traceback
		traceback.print_exc()


if __name__ == "__main__":
	import sys
	
	set_chinese_font()
	
	if len(sys.argv) > 1:
		mode = sys.argv[1]
	else:
		print("Please select run mode:")
		print("1. Train mode (train)")
		print("2. Test mode (test)")
		mode = input("Enter selection (train/test): ").strip().lower()
	
	if mode == 'train':
		main_train()
	elif mode == 'test':
		main_test()
	else:
		print("Invalid mode selection, please enter 'train' or 'test'")
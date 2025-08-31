from typing import Any, Dict, List
import os
import json
import time
from datetime import datetime
import cv2
import numpy as np

from benchmark.core.base_evaluator import BaseEvaluator
from benchmark.core.result_types import EvaluationResult


class BitwiseLogoEvaluator(BaseEvaluator):
	"""
	Evaluator for the Bitwise Logo Manipulation task.
	
	The agent must output three logo images with file names logo_output_1.png, 
	logo_output_2.png, and logo_output_3.png. We evaluate by:
	1. Generating target images using afondiel's bitwise logo manipulation algorithm
	2. Comparing output images pixel-by-pixel with target images
	3. Computing percentage of correctly matching pixels per image
	
	Expected output files: logo_output_1.png, logo_output_2.png, logo_output_3.png
	"""
	
	def __init__(self, config: Dict[str, Any]):
		self.config = config
		criteria = config.get("evaluation_criteria", {})
		self.pixel_match_threshold = float(criteria.get("pixel_match_threshold", 0.95))  # 95% pixels must match
		self.require_all_correct = criteria.get("require_all_correct", True)
		self.print_task_info()
	
	def _generate_target_image(self, logo_num: int, solution_folder: str) -> np.ndarray:
		"""Generate target image using the same algorithm as evaluation.py"""
		logo_file = os.path.join(solution_folder, f"logo_{logo_num}.png")
		background_file = os.path.join(solution_folder, "background.png")
		
		if not os.path.exists(logo_file) or not os.path.exists(background_file):
			raise FileNotFoundError(f"Required input files not found: {logo_file} or {background_file}")
		
		# Load logo image
		img_bgr = cv2.imread(logo_file)
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		logo_w = img_rgb.shape[0]
		logo_h = img_rgb.shape[1]
		
		# Load background image
		img_background_bgr = cv2.imread(background_file)
		img_background_rgb = cv2.cvtColor(img_background_bgr, cv2.COLOR_BGR2RGB)
		
		# Resize background to match logo dimensions while maintaining aspect ratio
		aspect_ratio = logo_w / img_background_rgb.shape[1]
		dim = (logo_w, int(img_background_rgb.shape[0] * aspect_ratio))
		img_background_rgb = cv2.resize(img_background_rgb, dim, interpolation=cv2.INTER_AREA)
		
		# Convert logo to grayscale and create binary mask
		img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
		retval, img_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
		
		# Create inverse mask
		img_mask_inv = cv2.bitwise_not(img_mask)
		
		# Create background behind logo lettering
		img_background = cv2.bitwise_and(img_background_rgb, img_background_rgb, mask=img_mask)
		
		# Isolate foreground using inverse mask
		img_foreground = cv2.bitwise_or(img_rgb, img_rgb, mask=img_mask_inv)
		
		# Combine background and foreground
		result = cv2.add(img_background, img_foreground)
		
		return result
	
	def _compare_images_pixel_by_pixel(self, output_img: np.ndarray, target_img: np.ndarray) -> float:
		"""Compare two images pixel by pixel and return percentage of matching pixels"""
		if output_img.shape != target_img.shape:
			return 0.0
		
		# Compare all pixels
		matches = np.all(output_img == target_img, axis=2)
		total_pixels = matches.size
		matching_pixels = np.sum(matches)
		
		return (matching_pixels / total_pixels) * 100.0
	
	def evaluate(self, solution_folder: str, solution_config: Any = None) -> EvaluationResult:
		start_time = time.time()
		task_id = self.config.get("task_id", "bitwise_logo")
		
		try:
			# Expected output files
			output_files = ["logo_output_1.png", "logo_output_2.png", "logo_output_3.png"]
			
			# Check if all output files exist
			missing_files = []
			for filename in output_files:
				output_path = os.path.join(solution_folder, filename)
				if not os.path.exists(output_path):
					missing_files.append(filename)
			
			if missing_files:
				return EvaluationResult(
					task_id=task_id,
					agent_id="unknown",
					timestamp=datetime.now(),
					metrics={},
					success=False,
					execution_time=time.time() - start_time,
					error_message=f"Missing output files: {', '.join(missing_files)}",
					artifacts={},
				)
			
			# Evaluate each logo
			per_image_accuracy: Dict[str, float] = {}
			total_accuracy = 0.0
			num_correct = 0
			
			for i in range(1, 4):
				output_file = f"logo_output_{i}.png"
				output_path = os.path.join(solution_folder, output_file)
				
				try:
					# Load output image
					output_img_bgr = cv2.imread(output_path)
					output_img_rgb = cv2.cvtColor(output_img_bgr, cv2.COLOR_BGR2RGB)
					
					# Generate target image
					target_img_rgb = self._generate_target_image(i, solution_folder)
					
					# Compare images pixel by pixel
					accuracy = self._compare_images_pixel_by_pixel(output_img_rgb, target_img_rgb)
					per_image_accuracy[output_file] = accuracy
					total_accuracy += accuracy
					
					# Check if this image passes the threshold
					if accuracy >= self.pixel_match_threshold * 100:
						num_correct += 1
						
				except Exception as e:
					per_image_accuracy[output_file] = 0.0
					print(f"Error processing {output_file}: {e}")
			
			avg_accuracy = total_accuracy / 3.0
			
			# Determine success
			if self.require_all_correct:
				success = (num_correct == 3)
			else:
				success_fraction_threshold = self.config.get("evaluation_criteria", {}).get("min_fraction_correct", 0.67)
				success = (num_correct / 3.0) >= success_fraction_threshold
			
			metrics = {
				"num_images": 3.0,
				"num_correct": float(num_correct),
				"avg_pixel_accuracy": float(avg_accuracy),
				"pixel_match_threshold": float(self.pixel_match_threshold * 100),
				"logo_1_accuracy": float(per_image_accuracy.get("logo_output_1.png", 0.0)),
				"logo_2_accuracy": float(per_image_accuracy.get("logo_output_2.png", 0.0)),
				"logo_3_accuracy": float(per_image_accuracy.get("logo_output_3.png", 0.0)),
			}
			
			artifacts = {
				"per_image_accuracy_json": json.dumps(per_image_accuracy),
			}
			
			return EvaluationResult(
				task_id=task_id,
				agent_id="unknown",
				timestamp=datetime.now(),
				metrics=metrics,
				success=success,
				execution_time=time.time() - start_time,
				error_message=None if success else f"Only {num_correct}/3 images meet accuracy threshold of {self.pixel_match_threshold*100:.1f}%",
				artifacts=artifacts,
			)
			
		except Exception as e:
			return EvaluationResult(
				task_id=task_id,
				agent_id="unknown",
				timestamp=datetime.now(),
				metrics={},
				success=False,
				execution_time=time.time() - start_time,
				error_message=f"Error during evaluation: {e}",
				artifacts={},
			)
	
	def get_metrics(self) -> List[str]:
		return [
			"num_images",
			"num_correct",
			"avg_pixel_accuracy",
			"pixel_match_threshold",
			"logo_1_accuracy",
			"logo_2_accuracy",
			"logo_3_accuracy",
		]
	
	def generate_report(self, results: List[EvaluationResult]) -> str:
		lines: List[str] = []
		for res in results:
			lines.append(f"Task: {res.task_id}, Success: {res.success}, Time: {res.execution_time:.2f}s")
			if res.error_message:
				lines.append(f"  Error: {res.error_message}")
			for k, v in res.metrics.items():
				if isinstance(v, float):
					if "accuracy" in k.lower():
						lines.append(f"  {k}: {v:.2f}%")
					else:
						lines.append(f"  {k}: {v:.4f}")
				else:
					lines.append(f"  {k}: {v}")
		return "\n".join(lines)

 

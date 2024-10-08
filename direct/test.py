import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from models import CifarModel, DriftModel
from utils import create_eval_dataset, compute_cdf, compute_p_cmm_test, predict_noise, fill_table_test

def main():
    base_input_dir = "../../dataset/gauss_rain"
    N = 1000
    p = 0

    # Load models
    cifar_model = CifarModel()
    checkpoint = torch.load("cifar10_model.ckpt", weights_only=True)
    cifar_model.load_state_dict(checkpoint['state_dict'])
    cifar_model.eval()

    drift_model = DriftModel()
    checkpoint = torch.load("noise_detection_model.ckpt", weights_only=True)
    drift_model.load_state_dict(checkpoint['state_dict'])
    drift_model.eval()

    # Load T(C,MG,MW), P(C,MG,MW)
    thresholds = pd.read_csv("./t_cmm/t_cmm_gauss_rain.csv")
    p_cmm = pd.read_csv("./p_cmm/p_cmm_gauss_rain.csv")

    # Select random noise condition
    noise_variances = np.arange(0.01, 0.20 + 0.01, 0.01)
    rain_lengths = np.arange(1, 20 + 1, 1)
    selected_noise_var = random.choice(noise_variances)
    selected_rain_len = random.choice(rain_lengths)

    # print(f"\n[INFO] Evaluating for random noise_var={selected_noise_var:.2f}, rain_len={selected_rain_len:02d}")

    eval_dataset = create_eval_dataset(base_input_dir, selected_noise_var, selected_rain_len, N, p)
    if eval_dataset is None:
        print("No dataset found for the selected noise and rain condition.")
        return

    eval_loader = DataLoader(eval_dataset, batch_size=256, shuffle=False, num_workers=8)
    
    # Drift detection - 노이즈가 감지된 이미지만 저장
    noisy_images = []
    
    with torch.inference_mode():
        # for images, _ in tqdm(eval_loader, desc="Detecting drift"):
        for images, _ in eval_loader:
            drift_outputs = drift_model(images)
            drift_predictions = torch.sigmoid(drift_outputs).round()

            for i in range(len(images)):
                if drift_predictions[i] == 1:
                    noisy_images.append(images[i].unsqueeze(0))
                    
    # print(f"Number of drifted images: {len(noisy_images)}")

    noisy_images = torch.cat(noisy_images, dim=0)
    noisy_loader = DataLoader(noisy_images, batch_size=256, shuffle=False, num_workers=8)
    
    # CDF F'(C)
    class_probabilities = {i: [] for i in range(10)}  # 클래스 수 10개
    with torch.inference_mode():
        # for images in tqdm(noisy_loader, desc="Processing images"):
        for images in noisy_loader:
            outputs = cifar_model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)

            for i in range(len(predicted_classes)):
                predicted_class = predicted_classes[i].item()
                predicted_prob = round(probabilities[i, predicted_class].item(), 3)
                class_probabilities[predicted_class].append(predicted_prob)

    cdfs = compute_cdf(class_probabilities)
    
    # P'(C,MG,MW)
    p_cmm_test = compute_p_cmm_test(cdfs, thresholds)
    p_cmm_test = fill_table_test(p_cmm_test)
    
    # Result
    predicted_noise_var, predicted_rain_len = predict_noise(p_cmm_test, p_cmm, selected_noise_var, selected_rain_len)
    # print(f"Predicted noise_var: {predicted_noise_var:.2f}, Predicted rain_len: {predicted_rain_len:02d}")

    return selected_noise_var, selected_rain_len, predicted_noise_var, predicted_rain_len

if __name__ == '__main__':
    main() 
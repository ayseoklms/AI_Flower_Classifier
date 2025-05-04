import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from multiprocessing import freeze_support 
from sklearn.metrics import precision_score, recall_score 

DATA_DIR = r"C:\Users\ayseo\OneDrive\Masaüstü\ai_project\data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "validation")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
MODEL_SAVE_PATH = "cicek_siniflandirici_resnet18.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Görüntü Dönüşümleri (Transforms) ---
train_transforms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

validation_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=15):
    """Modeli eğitir ve performans geçmişini döndürür."""
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {'train_loss': [], 'train_acc': [], 'train_precision': [], 'train_recall': [],
               'val_loss': [], 'val_acc': [], 'val_precision': [], 'val_recall': []}

    print("\nEğitim başlıyor..." + "\n" + "=" * 20)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}' + "\n" + '-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            dataloader = dataloaders[phase]
            dataset_size = len(dataloader.dataset)
            running_loss = 0.0
            running_corrects = 0
            all_labels_epoch = []
            all_preds_epoch = []

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels_on_device = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_on_device)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels_on_device.data)
                all_labels_epoch.extend(labels.cpu().numpy())
                all_preds_epoch.extend(preds.cpu().numpy())

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            epoch_precision = precision_score(all_labels_epoch, all_preds_epoch, average='macro', zero_division=0)
            epoch_recall = recall_score(all_labels_epoch, all_preds_epoch, average='macro', zero_division=0)

            print(f'{phase.capitalize()} Kayıp: {epoch_loss:.4f} Başarı: {epoch_acc:.4f} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f}')

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            history[f'{phase}_precision'].append(epoch_precision)
            history[f'{phase}_recall'].append(epoch_recall)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'*** Yeni en iyi doğrulama başarısı: {best_acc:.4f} (Epoch {epoch + 1}) ***')
        print()

    time_elapsed = time.time() - since
    print(f'Eğitim {time_elapsed // 60:.0f}dk {time_elapsed % 60:.0f}sn içinde tamamlandı.')
    print(f'En iyi Doğrulama Başarısı: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, history

# --- Grafik Çizdirme Fonksiyonu ---
def plot_history(history):
    """Eğitim/doğrulama için 4 metriği (Acc, Loss, Prec, Rec) çizer."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(history.get('train_acc', [])) + 1) # Güvenlik için .get 

    if not epochs:
        print("Uyarı: Çizilecek epoch verisi bulunamadı.")
        return

    metrics = [('Başarı (Accuracy)', 'acc'), ('Kayıp (Loss)', 'loss'),
               ('Precision', 'precision'), ('Recall', 'recall')]
    locations = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i, (title, key) in enumerate(metrics):
        row, col = locations[i]
        # Verilerin varlığını kontrol et
        train_key = f'train_{key}'
        val_key = f'val_{key}'
        if train_key in history and val_key in history:
             axs[row, col].plot(epochs, history[train_key], 'bo-', label=f'Eğitim {title}')
             axs[row, col].plot(epochs, history[val_key], 'ro-', label=f'Doğrulama {title}')
             axs[row, col].set_title(f'Model {title}')
             axs[row, col].set_ylabel(title)
             if row == 1:
                 axs[row, col].set_xlabel('Epoch')
             axs[row, col].legend()
             axs[row, col].grid(True)
        else:
            print(f"Uyarı: '{train_key}' veya '{val_key}' anahtarı geçmişte bulunamadı, {title} grafiği çizilemedi.")


    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    freeze_support()

    print("Kütüphaneler başarıyla yüklendi!")
    print(f"Kullanılacak Cihaz: {device}")

    try:
        print("\nVeri setleri yükleniyor...")
        train_dataset = ImageFolder(TRAIN_DIR, transform=train_transforms)
        validation_dataset = ImageFolder(VALID_DIR, transform=validation_transforms)
        print("Veri setleri başarıyla yüklendi.")

        class_names = train_dataset.classes
        num_classes = len(class_names)
        print(f"Bulunan Sınıflar ({num_classes} adet): {class_names}")
        idx_to_class = {i: name for i, name in enumerate(class_names)}
        print(f"Eğitim setindeki örnek sayısı: {len(train_dataset)}")
        print(f"Doğrulama setindeki örnek sayısı: {len(validation_dataset)}")
    except FileNotFoundError:
        print("-" * 30 + "\nHATA: Veri klasörleri bulunamadı!\n" + "-" * 30)
        print(f"Kontrol edilen yollar:\n Eğitim: {TRAIN_DIR}\n Doğrulama: {VALID_DIR}")
        print("Lütfen DATA_DIR değişkenini ve klasör yapısını kontrol edin.")
        exit()
    except Exception as e:
        print(f"Veri setleri yüklenirken bir hata oluştu: {e}")
        exit()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print("\nData Loader'lar hazırlandı.")
    print(f"Eğitim için batch sayısı: {len(train_loader)}")
    print(f"Doğrulama için batch sayısı: {len(validation_loader)}")

    print("\nModel yükleniyor ve ayarlanıyor...")
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    print(f"ResNet18 (Özellikler: {num_ftrs}, Sınıflar: {num_classes}) {device} cihazına taşındı.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("\nKayıp fonksiyonu: CrossEntropyLoss, Optimizer: Adam (lr=0.001)")

    print("\n--- Ana Program Başlıyor ---")
    dataloaders_dict = {'train': train_loader, 'val': validation_loader}

    model_trained, training_history = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataloaders=dataloaders_dict,
        device=device,
        num_epochs=NUM_EPOCHS
    )

    print("\nEğitim tamamlandı!")

    torch.save(model_trained.state_dict(), MODEL_SAVE_PATH)
    print(f"Eğitilmiş model şuraya kaydedildi: {MODEL_SAVE_PATH}")

    print("\nEğitim geçmişi grafikleri oluşturuluyor...")
  
    if training_history and all(isinstance(val_list, list) and len(val_list) > 0 for val_list in training_history.values()):
        plot_history(training_history)
    else:
        print("Uyarı: Eğitim geçmişi verisi eksik veya hatalı, grafik çizilemedi.")

    print("\nProgram başarıyla tamamlandı.")
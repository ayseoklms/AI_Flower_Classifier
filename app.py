import gradio as gr
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T
from PIL import Image 
import os

print("Gradio ve PyTorch kütüphaneleri yüklendi.")
MODEL_PATH = "cicek_siniflandirici_resnet18.pth" 
NUM_CLASSES = 5 
IMG_SIZE = 224 
DATA_DIR = r"C:\Users\ayseo\OneDrive\Masaüstü\ai_project\data"

class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

device = torch.device("cpu") 
print(f"Tahmin için kullanılacak cihaz: {device}")

preprocess_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print("Görüntü ön işleme dönüşümü tanımlandı.")

model = torchvision.models.resnet18()

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
print("Model mimarisi oluşturuldu (son katman ayarlandı).")

try:
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model ağırlıkları '{MODEL_PATH}' dosyasından başarıyla yüklendi.")
except FileNotFoundError:
    print(f"HATA: Model dosyası bulunamadı: {MODEL_PATH}")
    print("Lütfen train.py'nin başarıyla çalıştığından ve ßmodelin kaydedildiğinden emin olun.")
    exit()
except Exception as e:
    print(f"Model ağırlıkları yüklenirken bir hata oluştu: {e}")
    exit()

model.eval()
print("Model değerlendirme (eval) moduna alındı.")

model.to(device)
print(f"Model {device} cihazına taşındı.")

# --- Tahmin Fonksiyonu ---
def predict(input_image: Image.Image):
    """
    Verilen bir PIL görüntüsünü işler ve sınıf tahminini döndürür.

    Args:
        input_image: Kullanıcı tarafından yüklenen PIL Image nesnesi.

    Returns:
        dict: Sınıf isimlerini ve olasılıklarını içeren bir sözlük.
              Örn: {'rose': 0.85, 'tulip': 0.10, ...}
    """
    if input_image is None:
        return None

   
    img_tensor = preprocess_transform(input_image)
    img_tensor = img_tensor.unsqueeze(0) 
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0] 

    probs_cpu = probabilities.cpu().numpy()
    confidences = {class_names[i]: float(probs_cpu[i]) for i in range(NUM_CLASSES)}

    return confidences

print("Tahmin fonksiyonu tanımlandı.")
interface = gr.Interface(
    fn=predict,                
    inputs=gr.Image(type="pil", label="Çiçek Resmi Yükle"),
    outputs=gr.Label(num_top_classes=NUM_CLASSES, label="Tahmin Sonuçları"), 
    title="AI Destekli Çiçek Sınıflandırıcı",
    description="Bir çiçek resmi yükleyin ve modelin hangi tür olduğunu tahmin etmesini izleyin!",
    examples=[ # Kullanıcıların deneyebileceği örnek resimler 
        os.path.join(DATA_DIR, "validation/rose/15816400548_c551da2212_n.jpg"), 
        os.path.join(DATA_DIR, "validation/tulip/116343334_9cb4acdc57_n.jpg"), 
        os.path.join(DATA_DIR, "validation/sunflower/40410814_fba3837226_n.jpg") 
    ],
    allow_flagging="never" 
)

if __name__ == "__main__":
    print("\nGradio arayüzü başlatılıyor...")
    interface.launch() # link oluşturur
    print("Arayüz kapatıldı.")
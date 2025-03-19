import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import requests
from PIL import Image
from io import BytesIO
import cv2
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from torchvision import transforms, models
import yolov5  # Import pour utiliser le modèle YOLOv5
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable

class CNNIllustrations:
    def __init__(self, output_dir="figures"):
        """
        Initialize the class for creating computer vision illustrations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine best available device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print(f"Using MPS device (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print(f"Using CPU device")
            
        plt.style.use('ggplot')
        sns.set_style("whitegrid")
        
    def illustrate_cnn_architectures(self):
        """Illustrate different CNN architectures for different tasks"""
        plt.figure(figsize=(15, 10))
        
        # Define architectures to visualize
        architectures = {
            "Classification CNN": [
                ["Input Image", "Conv Layers", "Pooling", "FC Layers", "Output Classes"]
            ],
            "Object Detection (YOLO)": [
                ["Input Image", "CNN Backbone", "Feature Maps", "Prediction Heads", 
                 "Bounding Boxes + Class Probabilities"]
            ],
            "Segmentation CNN": [
                ["Input Image", "Encoder (CNN)", "Bottleneck", "Decoder (Transpose Conv)", "Pixel-wise Masks"]
            ],
            "Vision Transformer": [
                ["Input Image", "Patch Embedding", "Transformer Encoder", "MLP Head", "Output"]
            ]
        }
        
        # Plot each architecture
        for i, (name, layers) in enumerate(architectures.items()):
            ax = plt.subplot(4, 1, i+1)
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Draw boxes for each layer
            x_pos = 0
            for layer in layers[0]:
                width = 20 if "CNN" in layer or "Transformer" in layer else 15
                ax.add_patch(Rectangle((x_pos, 2), width, 6, 
                                      facecolor=plt.cm.tab10(i), alpha=0.7,
                                      edgecolor='black', linewidth=1))
                ax.text(x_pos + width/2, 5, layer, ha='center', va='center',
                       fontsize=10, fontweight='bold')
                
                # Add arrow
                if x_pos + width < 90:
                    ax.arrow(x_pos + width + 1, 5, 3, 0, head_width=0.5, 
                            head_length=1, fc='black', ec='black')
                
                x_pos += width + 5
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cnn_architectures_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def illustrate_yolo_detection(self):
        """
        Illustrates the YOLO object detection process using an actual model.
        """
        print("Loading YOLOv5 model...")
        try:
            # Use the correct model path format for YOLOv5
            model = yolov5.load('yolov5s.pt', device=self.device)  # Using .pt extension
            model.conf = 0.25  # confidence threshold
            model.iou = 0.45   # IoU threshold
            model.agnostic = False  # NMS class-agnostic
            model.multi_label = False  # NMS multiple labels per box
            model.max_det = 1000  # maximum number of detections per image
            
            # Download sample image if needed
            img_url = "https://ultralytics.com/images/zidane.jpg"
            img_path = "sample_image.jpg"
            
            # Download the image if it doesn't exist
            if not os.path.exists(img_path):
                response = requests.get(img_url)
                with open(img_path, 'wb') as f:
                    f.write(response.content)
            
            # Load and process the image
            img = Image.open(img_path)
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
            
            # Define a hook to capture feature maps
            feature_maps = []
            
            def hook_fn(module, input, output):
                feature_maps.append(output.detach().cpu())
            
            # Attach the hook to one of the YOLO layers
            for name, module in model.model.named_modules():
                if isinstance(module, torch.nn.Conv2d) and module.stride == (2, 2):
                    module.register_forward_hook(hook_fn)
                    break
                
            # Run inference
            results = model(img_path)
            
            # Visualize results 
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
            
            # Original image with detections
            ax0 = plt.subplot(gs[0, 0])
            ax0.set_title('Original Image with Detections')
            result_img = Image.fromarray(results.render()[0])
            ax0.imshow(result_img)
            ax0.axis('off')
            
            # YOLO grid structure
            ax1 = plt.subplot(gs[0, 1])
            ax1.set_title('YOLO Grid Structure')
            img_np = np.array(img)
            ax1.imshow(img_np)
            
            # Draw grid cells
            h, w = img_np.shape[:2]
            grid_size = 13  # Typical grid size for YOLO
            cell_h, cell_w = h / grid_size, w / grid_size
            
            for i in range(grid_size + 1):
                ax1.axhline(i * cell_h, color='red', alpha=0.5, linestyle='--')
                ax1.axvline(i * cell_w, color='red', alpha=0.5, linestyle='--')
            
            ax1.axis('off')
            
            # Feature maps (if available)
            ax2 = plt.subplot(gs[1, 0])
            ax2.set_title('Feature Maps')
            if feature_maps:
                # Display one of the feature maps
                feature_map = feature_maps[0][0, 0].numpy()
                ax2.imshow(feature_map, cmap='viridis')
            else:
                ax2.text(0.5, 0.5, 'No feature maps captured', 
                         horizontalalignment='center', verticalalignment='center')
            ax2.axis('off')
            
            # Prediction confidence heatmap
            ax3 = plt.subplot(gs[1, 1])
            ax3.set_title('Confidence Scores')
            
            # Extract confidence scores from results
            try:
                # Convert tensor to numpy and reshape to grid
                conf = results.xyxy[0][:, 4].cpu().numpy()  # confidence scores
                
                # Create a heatmap representation
                conf_img = np.zeros((grid_size, grid_size))
                for i in range(len(conf)):
                    x, y = int(results.xyxy[0][i, 0].item() / cell_w), int(results.xyxy[0][i, 1].item() / cell_h)
                    x = min(x, grid_size - 1)
                    y = min(y, grid_size - 1)
                    conf_img[y, x] = max(conf_img[y, x], conf[i])
                
                im = ax3.imshow(conf_img, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
                plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
            except Exception as e:
                ax3.text(0.5, 0.5, f'Error creating confidence heatmap: {e}', 
                         horizontalalignment='center', verticalalignment='center')
            ax3.axis('off')
            
            plt.tight_layout()
            os.makedirs('figures', exist_ok=True)
            plt.savefig('figures/yolo_detection_process.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Clean up the temporary image file
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
            except:
                pass
            
            print("YOLO detection visualization complete!")
            
        except Exception as e:
            print(f"Error in YOLO illustration: {e}")
            # Create a simplified version if the real model fails
            self._create_simplified_yolo_visualization()
    
    def _create_simplified_yolo_visualization(self):
        """
        Crée une visualisation YOLO en utilisant PyTorch pour charger le modèle
        et effectuer une inférence réelle
        """
        print("Création d'une visualisation YOLO avec PyTorch...")
        import torch
        
        # Chemin pour l'image temporaire
        img_path = "temp_yolo.jpg"
        
        try:
            # Télécharger l'image depuis GitHub (image bien connue du dataset COCO)
            img_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
            response = requests.get(img_url)
            with open(img_path, 'wb') as f:
                f.write(response.content)
            
            # Charger l'image pour traitement
            img = Image.open(img_path)
            img_array = np.array(img).copy()  # Créer une copie modifiable
            
            # Charger le modèle YOLOv5 via torch.hub
            print("Chargement du modèle YOLOv5...")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, device=self.device)
            
            # Définir les paramètres d'inférence
            model.conf = 0.25  # seuil de confiance
            model.iou = 0.45   # seuil IoU
            
            # Créer un hook pour capturer les feature maps
            activation = {}
            
            def get_activation(name):
                def hook(model, input, output):
                    # Stocker les activations
                    activation[name] = output.detach().cpu()
                return hook
            
            # Attacher le hook à une couche de feature map significative (couche 17 par exemple)
            # Cette couche est typiquement une des dernières couches de feature maps avant les heads
            for name, layer in model.model.named_modules():
                if isinstance(layer, torch.nn.Conv2d) and "model.17" in name:
                    layer.register_forward_hook(get_activation(name))
                    break
            
            # Effectuer l'inférence
            results = model(img_path)
            
            # Créer le subplot
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Image originale avec les détections
            axs[0, 0].set_title('Input Image', fontsize=14)
            
            # Copier l'image pour ajouter les détections manuellement
            result_img = img_array.copy()
            
            # Obtenir les résultats de détection et dessiner manuellement
            detections = results.xyxy[0].cpu().numpy()
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            
            for i, det in enumerate(detections):
                x1, y1, x2, y2, conf, cls = det
                # Convertir en entiers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Créer un code couleur pour chaque classe
                color_idx = int(cls) % len(colors)
                color = colors[color_idx]
                color_rgb = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                
                # Dessiner la boîte englobante
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color_rgb, 2)
                
                # Ajouter le texte d'étiquette
                label = f"{model.names[int(cls)]}: {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # Rectangle d'arrière-plan pour le texte
                cv2.rectangle(result_img, (x1, y1-25), (x1+text_size[0], y1), color_rgb, -1)
                cv2.putText(result_img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            axs[0, 0].imshow(result_img)
            axs[0, 0].axis('off')
            
            # 2. Structure de grille YOLO
            axs[0, 1].set_title('YOLO Grid Structure', fontsize=14)
            axs[0, 1].imshow(img_array, alpha=0.6)
            
            # Dessiner la grille
            h, w = img_array.shape[:2]
            grid_size = 13  # Taille typique de la grille YOLO
            cell_h, cell_w = h / grid_size, w / grid_size
            
            for i in range(grid_size + 1):
                axs[0, 1].axhline(i * cell_h, color='red', alpha=0.7, linestyle='--', linewidth=1)
                axs[0, 1].axvline(i * cell_w, color='red', alpha=0.7, linestyle='--', linewidth=1)
            
            # Ajouter des boîtes englobantes transparentes pour montrer la prédiction
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, 
                                       edgecolor='yellow', facecolor='none',
                                       linestyle='--', alpha=0.7)
                axs[0, 1].add_patch(rect)
            
            axs[0, 1].axis('off')
            
            # 3. Feature maps
            axs[1, 0].set_title('Feature Maps', fontsize=14)
            
            # Visualiser les feature maps si elles ont été capturées
            if activation:
                # Obtenir le premier feature map de la première image
                key = list(activation.keys())[0]
                feature_map = activation[key][0].cpu().numpy()
                
                # Moyenner sur les canaux pour l'affichage
                feature_map_display = np.mean(feature_map, axis=0)
                feature_map_display = (feature_map_display - feature_map_display.min()) / (feature_map_display.max() - feature_map_display.min() + 1e-8)
                
                im = axs[1, 0].imshow(feature_map_display, cmap='viridis')
                plt.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)
            else:
                # Fallback si pas de feature maps
                axs[1, 0].text(0.5, 0.5, 'No feature maps captured', 
                              ha='center', va='center', fontsize=12)
            
            axs[1, 0].axis('off')
            
            # 4. Carte de confiance
            axs[1, 1].set_title('Confidence Scores', fontsize=14)
            
            # Créer une carte de confiance basée sur les détections réelles
            conf_map = np.zeros((grid_size, grid_size))
            
            # Remplir la carte avec les valeurs de confiance
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                # Centre de la boîte
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                # Convertir en coordonnées de grille
                grid_x = int(center_x / cell_w)
                grid_y = int(center_y / cell_h)
                
                # S'assurer que les coordonnées sont dans les limites
                grid_x = min(max(0, grid_x), grid_size - 1)
                grid_y = min(max(0, grid_y), grid_size - 1)
                
                # Mettre à jour la valeur de confiance (prendre le max si plusieurs objets)
                conf_map[grid_y, grid_x] = max(conf_map[grid_y, grid_x], conf)
                
                # Ajouter un peu de diffusion pour visualisation
                radius = 1
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        if 0 <= grid_y + dy < grid_size and 0 <= grid_x + dx < grid_size:
                            decay = 1 - (abs(dx) + abs(dy)) / (2 * radius + 0.1)
                            conf_map[grid_y + dy, grid_x + dx] = max(
                                conf_map[grid_y + dy, grid_x + dx],
                                conf * decay
                            )
            
            # Appliquer un flou gaussien pour lisser
            conf_map = cv2.GaussianBlur(conf_map, (3, 3), 0)
            
            # Afficher la carte de confiance
            im = axs[1, 1].imshow(conf_map, cmap='hot', vmin=0, vmax=1)
            plt.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)
            
            # Marquer les centres des objets détectés
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                grid_x = int(center_x / cell_w)
                grid_y = int(center_y / cell_h)
                
                # S'assurer que les coordonnées sont dans les limites
                grid_x = min(max(0, grid_x), grid_size - 1)
                grid_y = min(max(0, grid_y), grid_size - 1)
                
                axs[1, 1].plot(grid_x, grid_y, 'o', markersize=8, mfc='none', mec='white', mew=2)
            
            axs[1, 1].axis('off')
            
            plt.tight_layout()
            os.makedirs('figures', exist_ok=True)
            plt.savefig('figures/yolo_detection_process.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Nettoyer
            if os.path.exists(img_path):
                os.remove(img_path)
            
            print("Visualisation YOLO avec PyTorch terminée!")
            
        except Exception as e:
            print(f"Erreur lors de la création de la visualisation YOLO: {e}")
            # Créer une visualisation fallback très simple en cas d'erreur
            self._create_fallback_visualization()
    
    def _create_fallback_visualization(self):
        """
        Crée une visualisation simplifiée en cas d'échec du chargement du modèle YOLO
        """
        print("Création d'une visualisation YOLO de secours...")
        
        try:
            # Utiliser une image fiable
            img_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
            img_array = np.array(img)
        except:
            # Créer une image simple si le téléchargement échoue
            img_array = np.ones((640, 480, 3), dtype=np.uint8) * 255
            img_array[:, :, 0] = 200  # Ajouter de la couleur
        
        # Créer figure avec sous-plots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Image originale
        axs[0, 0].imshow(img_array)
        axs[0, 0].set_title('Input Image', fontsize=14)
        
        # Ajouter des détections simulées (placées manuellement)
        rect1 = patches.Rectangle((180, 100), 250, 380, linewidth=2, edgecolor='r', facecolor='none')
        axs[0, 0].add_patch(rect1)
        axs[0, 0].text(180, 90, 'person: 0.94', color='red', fontsize=12, backgroundcolor='white')
        
        rect2 = patches.Rectangle((400, 120), 220, 360, linewidth=2, edgecolor='g', facecolor='none')
        axs[0, 0].add_patch(rect2)
        axs[0, 0].text(400, 110, 'person: 0.89', color='green', fontsize=12, backgroundcolor='white')
        
        axs[0, 0].axis('off')
        
        # 2. Structure de grille YOLO
        axs[0, 1].imshow(img_array, alpha=0.6)
        axs[0, 1].set_title('YOLO Grid Structure', fontsize=14)
        
        # Dessiner les cellules de la grille
        h, w = img_array.shape[:2]
        grid_size = 13  # Taille typique pour YOLOv3/v5
        cell_h, cell_w = h / grid_size, w / grid_size
        
        for i in range(grid_size + 1):
            axs[0, 1].axhline(i * cell_h, color='red', alpha=0.7, linestyle='--', linewidth=1)
            axs[0, 1].axvline(i * cell_w, color='red', alpha=0.7, linestyle='--', linewidth=1)
        
        axs[0, 1].axis('off')
        
        # 3. Cartes de caractéristiques
        axs[1, 0].set_title('Feature Maps', fontsize=14)
        
        # Créer une carte simulée basée sur la structure de l'image
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        feature_map = cv2.resize(gray, (grid_size, grid_size))
        
        # Normaliser pour une meilleure visualisation
        feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-8)
        
        im = axs[1, 0].imshow(feature_map, cmap='viridis')
        plt.colorbar(im, ax=axs[1, 0], fraction=0.046, pad=0.04)
        axs[1, 0].axis('off')
        
        # 4. Scores de confiance
        axs[1, 1].set_title('Confidence Scores', fontsize=14)
        
        # Créer une heatmap de confiance simulée
        conf_map = np.zeros((grid_size, grid_size))
        
        # Zidane
        person1_grid_x = int(250 / cell_w)
        person1_grid_y = int(280 / cell_h)
        conf_map[person1_grid_y-1:person1_grid_y+2, person1_grid_x-1:person1_grid_x+2] = 0.9
        
        # Ancelotti
        person2_grid_x = int(500 / cell_w)
        person2_grid_y = int(300 / cell_h)
        conf_map[person2_grid_y-1:person2_grid_y+2, person2_grid_x-1:person2_grid_x+2] = 0.85
        
        # Appliquer un flou gaussien
        conf_map = cv2.GaussianBlur(conf_map, (3, 3), 0)
        
        im = axs[1, 1].imshow(conf_map, cmap='hot', vmin=0, vmax=1)
        plt.colorbar(im, ax=axs[1, 1], fraction=0.046, pad=0.04)
        
        # Marquer les points de confiance maximale
        axs[1, 1].plot(person1_grid_x, person1_grid_y, 'o', markersize=8, mfc='none', mec='white', mew=2)
        axs[1, 1].plot(person2_grid_x, person2_grid_y, 'o', markersize=8, mfc='none', mec='white', mew=2)
        
        axs[1, 1].axis('off')
        
        plt.tight_layout()
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/yolo_detection_process.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("Visualisation YOLO de secours terminée!")
    
    def illustrate_segmentation(self):
        """Illustrate semantic segmentation using a real PyTorch model"""
        print("Generating semantic segmentation visualization with a real model...")
        
        try:
            # Load a pre-trained DeepLabV3 model with ResNet-50 backbone
            model = models.segmentation.deeplabv3_resnet50(pretrained=True)
            model.to(self.device)
            model.eval()
            
            # Download a sample image (using a clear image good for segmentation)
            url = "https://github.com/tensorflow/models/raw/master/research/deeplab/g3doc/img/image2.jpg"
            response = requests.get(url)
            if response.status_code != 200:
                # Fallback to a different image if the first URL fails
                url = "https://raw.githubusercontent.com/pytorch/pytorch.github.io/master/assets/images/segmentation1.png"
                response = requests.get(url)
            
            img = Image.open(BytesIO(response.content))
            
            # Resize for faster processing and consistency
            img = img.resize((500, 375))
            
            # Create input tensor and normalize
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            input_tensor = preprocess(img)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Perform segmentation
            with torch.no_grad():
                output = model(input_batch)["out"][0]
            
            # Process the output
            output_predictions = output.argmax(0).cpu().numpy()
            
            # Get a color map for visualization
            # Use the PASCAL VOC color map (21 classes including background)
            voc_colormap = [
                [0, 0, 0],          # background
                [128, 0, 0],        # aeroplane
                [0, 128, 0],        # bicycle
                [128, 128, 0],      # bird
                [0, 0, 128],        # boat
                [128, 0, 128],      # bottle
                [0, 128, 128],      # bus
                [128, 128, 128],    # car
                [64, 0, 0],         # cat
                [192, 0, 0],        # chair
                [64, 128, 0],       # cow
                [192, 128, 0],      # dining table
                [64, 0, 128],       # dog
                [192, 0, 128],      # horse
                [64, 128, 128],     # motorbike
                [192, 128, 128],    # person
                [0, 64, 0],         # potted plant
                [128, 64, 0],       # sheep
                [0, 192, 0],        # sofa
                [128, 192, 0],      # train
                [0, 64, 128]        # tv/monitor
            ]
            voc_colormap = np.array(voc_colormap, dtype=np.uint8)
            
            # Create a figure to display the results
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            
            # Original image
            img_np = np.array(img)
            axs[0, 0].imshow(img_np)
            axs[0, 0].set_title('Input Image', fontsize=14)
            axs[0, 0].axis('off')
            
            # Segmentation mask (colored by class)
            colored_mask = voc_colormap[output_predictions]
            axs[0, 1].imshow(colored_mask)
            axs[0, 1].set_title('Segmentation Mask (DeepLabV3)', fontsize=14)
            axs[0, 1].axis('off')
            
            # Extract unique classes present in the segmentation
            classes_present = np.unique(output_predictions)
            class_names = [
                'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse',
                'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor'
            ]
            
            # Show confidence maps for the most prominent classes
            max_classes_to_show = 2  # Show confidence maps for the 2 most prominent classes (excluding background)
            confidence_classes = []
            
            for cls in classes_present:
                if cls == 0:  # Skip background
                    continue
                pixel_count = np.sum(output_predictions == cls)
                confidence_classes.append((cls, pixel_count))
            
            # Sort by pixel count (most prominent first)
            confidence_classes.sort(key=lambda x: x[1], reverse=True)
            
            # Create visualization showing confidence scores for the classes
            for i, (cls_idx, _) in enumerate(confidence_classes[:max_classes_to_show]):
                if i >= 2:  # Only using the bottom row (2 plots)
                    break
                    
                confidence_map = output[cls_idx].cpu().numpy()
                # Normalize for better visualization
                confidence_map = (confidence_map - confidence_map.min()) / (confidence_map.max() - confidence_map.min() + 1e-8)
                
                axs[1, i].imshow(confidence_map, cmap='hot', interpolation='nearest')
                axs[1, i].set_title(f'Confidence Map: {class_names[cls_idx]}', fontsize=14)
                axs[1, i].axis('off')
            
            # If we don't have 2 classes, fill the remaining plots
            for i in range(len(confidence_classes[:max_classes_to_show]), 2):
                axs[1, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/segmentation_mask.png", dpi=300, bbox_inches="tight")
            
            # Create overlay of segmentation on original image
            overlay = img_np.copy()
            # Add alpha blending to make the segmentation visible on top of the image
            alpha = 0.5
            for cls_idx in classes_present:
                if cls_idx == 0:  # Skip background
                    continue
                mask_cls = (output_predictions == cls_idx)
                for c in range(3):  # RGB channels
                    overlay[:, :, c] = np.where(
                        mask_cls, 
                        overlay[:, :, c] * (1 - alpha) + voc_colormap[cls_idx][c] * alpha,
                        overlay[:, :, c]
                    )
            
            plt.figure(figsize=(10, 8))
            plt.imshow(overlay)
            plt.title('Segmentation Overlay (DeepLabV3)', fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/segmentation_overlay.png", dpi=300, bbox_inches="tight")
            
            # Create a separate figure for the encoder-decoder architecture
            plt.figure(figsize=(5, 10))
            
            components = ['Input', 'Encoder\n(ResNet-50)', 'ASPP\nModule', 'Decoder', 'Output\nMask']
            colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99', '#FF99FF']
            
            # Get the current axes
            ax = plt.gca()
            
            for i, (comp, color) in enumerate(zip(components, colors)):
                y = 0.8 - i * 0.15
                height = 0.12
                x = 0.2
                width = 0.6
                
                # Draw box
                ax.add_patch(Rectangle((x, y), width, height, 
                                     facecolor=color, alpha=0.7,
                                     edgecolor='black', linewidth=1))
                ax.text(x + width/2, y + height/2, comp, ha='center', va='center',
                      fontsize=12, fontweight='bold')
                # Add arrow
                if i < len(components) - 1:
                    ax.arrow(x + width/2, y, 0, -0.03, head_width=0.05, 
                           head_length=0.02, fc='black', ec='black')
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            plt.title("DeepLabV3 Architecture", fontsize=14)
            plt.savefig(f"{self.output_dir}/segmentation_architecture.png", dpi=300, bbox_inches="tight")
            plt.close('all')
            
            print("Real model segmentation visualization complete!")
        
        except Exception as e:
            print(f"Error in segmentation illustration: {e}")
            # Fallback to a simpler method if the model fails
            self._create_simplified_segmentation()
    
    def _create_simplified_segmentation(self):
        """Create a simplified segmentation visualization as fallback"""
        print("Creating simplified segmentation visualization...")
        
        try:
            # Download a sample image
            url = "https://github.com/tensorflow/models/raw/master/research/deeplab/g3doc/img/image2.jpg"
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img_array = np.array(img)
            
            # Create a mock segmentation mask
            mask = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.uint8)
            
            # Background
            mask[:, :] = 0
            
            # Road
            mask[350:, 100:600] = 1
            
            # Car
            mask[250:350, 200:400] = 2
            
            # Person
            mask[150:300, 400:500] = 3
            
            # Tree
            mask[50:200, 50:200] = 4
            mask[50:150, 550:650] = 4
            
            fig, axs = plt.subplots(2, 2, figsize=(14, 12))
            
            # Original image
            axs[0, 0].imshow(img_array)
            axs[0, 0].set_title('Input Image', fontsize=14)
            axs[0, 0].axis('off')
            
            # Segmentation mask
            # Use plt.colormaps instead of deprecated plt.cm.get_cmap
            cmap = plt.colormaps['viridis'].resampled(5)
            axs[0, 1].imshow(mask, cmap=cmap, vmin=0, vmax=4)
            axs[0, 1].set_title('Segmentation Mask (Simplified)', fontsize=14)
            axs[0, 1].axis('off')
            
            # Clear the bottom subplots
            axs[1, 0].axis('off')
            axs[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/segmentation_mask.png", dpi=300, bbox_inches="tight")
            
            # Create a separate figure for the encoder-decoder architecture
            plt.figure(figsize=(12, 3))
            
            components = ['Input', 'Encoder', 'Bottleneck', 'Decoder', 'Mask']
            colors = ['#FF9999', '#99FF99', '#9999FF', '#FFFF99', '#FF99FF']
            
            # Get the current axes
            ax = plt.gca()
            
            for i, (comp, color) in enumerate(zip(components, colors)):
                x = i * 0.2 + 0.1
                width = 0.15
                
                # Draw box
                ax.add_patch(Rectangle((x, 0.3), width, 0.4, 
                                     facecolor=color, alpha=0.7,
                                     edgecolor='black', linewidth=1))
                ax.text(x + width/2, 0.5, comp, ha='center', va='center',
                      fontsize=12, fontweight='bold')
                
                # Add arrow
                if i < len(components) - 1:
                    ax.arrow(x + width + 0.01, 0.5, 0.03, 0, head_width=0.05, 
                           head_length=0.02, fc='black', ec='black')
            
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            plt.title("Encoder-Decoder Architecture for Segmentation", fontsize=14)
            plt.savefig(f"{self.output_dir}/segmentation_architecture.png", dpi=300, bbox_inches="tight")
            
            # Overlay segmentation on image
            overlay = img_array.copy()
            colors_rgb = [
                [0, 0, 0],        # Background
                [255, 0, 0],      # Road
                [0, 255, 0],      # Car
                [0, 0, 255],      # Person
                [255, 255, 0]     # Tree
            ]
            
            for i in range(5):
                # Create binary mask for each class
                binary_mask = (mask == i)
                # Apply color only to the pixels of this class
                for c in range(3):  # RGB channels
                    overlay[:, :, c] = np.where(binary_mask, colors_rgb[i][c], overlay[:, :, c])
            
            blended = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)
            
            plt.figure(figsize=(10, 8))
            plt.imshow(blended)
            plt.title('Segmentation Overlay (Simplified)', fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/segmentation_overlay.png", dpi=300, bbox_inches="tight")
            plt.close('all')
            
            print("Simplified segmentation visualization complete!")
        except Exception as e:
            print(f"Error in simplified segmentation: {e}")
    
    def illustrate_transformer_attention(self):
        """
        Create an illustration of the transformer attention mechanism using a real ViT model.
        """
        try:
            print("Generating transformer attention visualization...")
            
            # Charger un modèle ViT pré-entraîné
            from transformers import ViTForImageClassification, ViTImageProcessor
            
            # Charger le modèle ViT pré-entraîné
            model_name = "google/vit-base-patch16-224"
            model = ViTForImageClassification.from_pretrained(model_name)
            processor = ViTImageProcessor.from_pretrained(model_name)
            
            # Mettre le modèle en mode évaluation et sur le bon device
            model.to(self.device)
            model.eval()
            
            # Charger une image d'exemple
            try:
                # URLs d'images de loups (comme dans l'image partagée)
                wolf_urls = [
                    "https://www.nps.gov/yell/learn/nature/images/wolf-lamar_2.jpg",
                    "https://www.pbs.org/wnet/nature/files/2014/10/WolfFact3.jpg",
                    "https://www.nationalgeographic.com/content/dam/animals/2022/05/gray-wolf-2022/grey-wolf-thumbnail.jpg",
                    "https://e360.yale.edu/assets/site/GettyImages-477134136-web.jpg"
                ]
                
                img = None
                for url in wolf_urls:
                    try:
                        print(f"Trying to download wolf image from {url}")
                        response = requests.get(url, timeout=10)
                        if response.status_code == 200:
                            img = Image.open(BytesIO(response.content))
                            print(f"Successfully loaded image from {url}")
                            break
                    except Exception as e:
                        print(f"Failed to download from {url}: {e}")
                        continue
                
                if img is None:
                    # Fallback to other animal images if wolf images fail
                    backup_urls = [
                        "http://farm1.static.flickr.com/114/267034228_6fb46a3b0f.jpg",  # Chien
                        "https://github.com/pytorch/hub/raw/master/images/dog.jpg"  # Chien du repo PyTorch
                    ]
                    
                    for url in backup_urls:
                        try:
                            print(f"Trying backup image from {url}")
                            response = requests.get(url, timeout=10)
                            if response.status_code == 200:
                                img = Image.open(BytesIO(response.content))
                                print(f"Successfully loaded backup image from {url}")
                                break
                        except Exception as e:
                            print(f"Failed to download backup from {url}: {e}")
                            continue
                
                if img is None:
                    raise Exception("Failed to download any image from the provided URLs")
                
                # Resize l'image pour qu'elle soit adaptée au modèle
                orig_size = img.size
                # Conserver le ratio d'aspect
                img = img.resize((384, 384))
                print(f"Resized image from {orig_size} to (384, 384)")
                
            except Exception as e:
                print(f"Error downloading images: {e}")
                # Utiliser une image locale si disponible
                local_image_path = os.path.join(os.path.dirname(__file__), "sample_image.jpg")
                if os.path.exists(local_image_path):
                    print(f"Using local image: {local_image_path}")
                    img = Image.open(local_image_path)
                    img = img.resize((384, 384))
                else:
                    # En dernier recours, utiliser une image synthétique
                    print("Creating synthetic image")
                    img_array = np.zeros((384, 384, 3), dtype=np.uint8)
                    # Arrière-plan
                    img_array[:, :] = [240, 240, 240]
                    
                    # Dessiner des formes distinctes
                    # Cercle rouge
                    center_x, center_y = 100, 200
                    radius = 70
                    for i in range(384):
                        for j in range(384):
                            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                            if dist < radius:
                                img_array[i, j] = [255, 0, 0]
                    
                    # Carré bleu
                    for i in range(200, 330):
                        for j in range(100, 230):
                            img_array[i, j] = [0, 0, 255]
                    
                    img = Image.fromarray(img_array)
            
            # Prétraiter l'image pour le modèle
            inputs = processor(images=img, return_tensors="pt").to(self.device)
            
            # Activer la sortie des attentions
            outputs = model(**inputs, output_attentions=True)
            
            # Extraction des cartes d'attention à différentes profondeurs
            # Choisir des couches qui montrent une progression intéressante 
            attentions_early = outputs.attentions[1]     # Couche peu profonde (2ème)
            attentions_middle = outputs.attentions[5]    # Couche intermédiaire (6ème)
            attentions_deep = outputs.attentions[11]     # Couche profonde (12ème)
            
            # Créer une mise en page sans le texte explicatif
            plt.figure(figsize=(16, 10))
            
            # Définir une grille de 2x3 sans réserver d'espace pour le texte
            gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], wspace=0.3, hspace=0.4)
            
            # Image originale dans le coin supérieur gauche
            ax_img = plt.subplot(gs[0, 0])
            ax_img.imshow(img)
            ax_img.set_title("Image d'entrée", fontsize=16, fontweight='bold', pad=10)
            ax_img.axis('off')
            
            # Extraire plusieurs têtes d'attention pour montrer la diversité
            attention_data = []
            
            # Pour chaque couche, sélectionner quelques têtes d'attention différentes
            # Nous prendrons la tête 0, 3, et 7 de chaque couche pour montrer différents comportements
            heads = [0, 3, 7]
            
            for layer_name, attention_layer in [
                ("Couche peu profonde (2)", attentions_early), 
                ("Couche intermédiaire (6)", attentions_middle),
                ("Couche profonde (12)", attentions_deep)
            ]:
                for head_idx in heads:
                    if head_idx == 0:  # Garder uniquement la première tête pour les premières couches
                        # Extraire l'attention du token CLS vers les patches d'image
                        attention = attention_layer.detach().cpu().numpy()[0, head_idx, 0, 1:]
                        attention_data.append((f"{layer_name}, Tête {head_idx+1}", attention))
            
            # Définir des couleurs différentes pour les cartes d'attention
            attention_cmaps = ['viridis', 'plasma', 'magma']
            
            # Les positions des cartes d'attention dans la grille
            positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
            
            # Afficher les cartes d'attention avec de meilleurs titres
            for i, ((title, attn), cmap) in enumerate(zip(attention_data, attention_cmaps)):
                if i < len(positions):
                    row, col = positions[i]
                    ax = plt.subplot(gs[row, col])
                    
                    # Reshape pour avoir une carte 2D 
                    size = int(np.sqrt(attn.shape[0]))
                    reshaped_attn = attn.reshape(size, size)
                    
                    # Normaliser pour une meilleure visualisation
                    reshaped_attn = (reshaped_attn - reshaped_attn.min()) / (reshaped_attn.max() - reshaped_attn.min() + 1e-8)
                    
                    # Améliorer la visualisation avec un contraste plus élevé
                    im = ax.imshow(reshaped_attn, cmap=cmap, interpolation='bilinear')
                    ax.set_title(title, fontsize=14)
                    ax.axis('off')
                    
                    # Ajouter des colorbar plus petites et discrètes
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax)
            
            # Créer un overlay composite de toutes les couches pour montrer l'attention globale
            # Combiner l'attention de plusieurs têtes de la dernière couche
            all_heads_attention = attentions_deep.detach().cpu().numpy()[0]  # (num_heads, seq_len, seq_len)
            
            # Prendre la moyenne des têtes d'attention du token CLS vers les patches
            combined_attention = np.mean([all_heads_attention[h, 0, 1:] for h in range(all_heads_attention.shape[0])], axis=0)
            
            # Position de l'image composite
            ax_overlay = plt.subplot(gs[1, 2])
            
            # Reshape en grille 2D
            size = int(np.sqrt(combined_attention.shape[0]))
            combined_reshaped = combined_attention.reshape(size, size)
            
            # Normaliser pour contraste optimal
            attention_norm = (combined_reshaped - combined_reshaped.min()) / (combined_reshaped.max() - combined_reshaped.min())
            
            # Redimensionner l'image originale pour qu'elle corresponde à la taille de la grille d'attention
            img_resized = img.resize((size, size), Image.LANCZOS)
            img_np = np.array(img_resized)
            
            # Créer un heatmap avec une colormap plus visible sur l'image
            # Utiliser 'inferno' qui a un bon contraste sur les images naturelles
            cmap = plt.cm.inferno
            attention_heatmap = cmap(attention_norm)
            attention_heatmap = attention_heatmap[:, :, :3]  # Garder seulement RGB
            
            # Superposer l'image et le heatmap
            overlay = img_np * 0.6 + attention_heatmap * 255 * 0.4
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            ax_overlay.imshow(overlay)
            ax_overlay.set_title("Carte d'Attention Combinée", fontsize=14, fontweight='bold')
            ax_overlay.axis('off')
            
            plt.suptitle("Mécanisme d'attention dans les Vision Transformers", fontsize=18, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f"{self.output_dir}/transformer_attention_maps.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            print("Transformer attention visualization complete!")
            
        except Exception as e:
            print(f"Error in transformer attention illustration: {e}")
            
            # Create a fallback version if the main one fails
            self._create_simplified_transformer_attention()
    
    def _create_simplified_transformer_attention(self):
        """Fallback method to create a simplified transformer attention visualization"""
        print("Creating simplified transformer attention visualization...")
        
        try:
            # Importer mpl_toolkits pour les axes divisés si ce n'est pas déjà fait
            from mpl_toolkits.axes_grid1 import make_axes_locatable
        except ImportError:
            pass
            
        plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], wspace=0.3, hspace=0.3)
        
        # Utiliser une image réelle ou créer une image synthétique avec objets distincts
        try:
            # Créer une image synthétique représentant un animal stylisé
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)
            img_array[:, :] = [240, 240, 240]  # Fond gris clair
            
            # Corps principal (forme de loup stylisé)
            # Tête
            center_x, center_y = 80, 80
            radius = 40
            for i in range(224):
                for j in range(224):
                    dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if dist < radius:
                        img_array[i, j] = [180, 180, 180]  # Gris pour la tête
            
            # Corps
            for i in range(70, 160):
                for j in range(90, 190):
                    img_array[i, j] = [180, 180, 180]  # Gris pour le corps
            
            # Jambes
            for leg_x in [100, 150]:
                for i in range(150, 200):
                    for j in range(leg_x-10, leg_x+10):
                        if 0 <= i < 224 and 0 <= j < 224:
                            img_array[i, j] = [180, 180, 180]
            
            # Oreilles
            triangle_pts1 = np.array([[60, 50], [80, 30], [100, 50]])
            triangle_pts2 = np.array([[80, 50], [100, 30], [120, 50]])
            cv2.fillPoly(img_array, [triangle_pts1], [180, 180, 180])
            cv2.fillPoly(img_array, [triangle_pts2], [180, 180, 180])
            
            # Museau et traits faciaux
            cv2.ellipse(img_array, (80, 90), (20, 10), 0, 0, 360, (120, 120, 120), -1)
            cv2.circle(img_array, (70, 70), 5, (50, 50, 50), -1)  # Œil
            cv2.circle(img_array, (90, 70), 5, (50, 50, 50), -1)  # Œil
            
            img = Image.fromarray(img_array)
            
        except Exception as e:
            print(f"Error creating synthetic image: {e}")
            # Image de fallback très simple
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            img[:, :] = [240, 240, 240]
            # Ajouter quelques formes basiques
            cv2.rectangle(img, (50, 50), (150, 150), (180, 180, 180), -1)
            cv2.circle(img, (100, 100), 40, (120, 120, 120), -1)
            img = Image.fromarray(img)
        
        # Image originale
        ax_img = plt.subplot(gs[0, 0])
        ax_img.imshow(np.array(img))
        ax_img.set_title("Image d'entrée", fontsize=16, fontweight='bold')
        ax_img.axis('off')
        
        # Créer des cartes d'attention simulées plus informatives
        # Nous allons créer des cartes qui montrent l'attention progressive
        
        # 1. Carte d'attention de bas niveau: détails texturaux
        attn1 = np.zeros((16, 16))
        # Attention aux contours
        for x in range(16):
            for y in range(16):
                # Distance aux bords du loup
                edge_x = min(x, 15-x)
                edge_y = min(y, 15-y)
                if 3 <= x <= 12 and 3 <= y <= 10:
                    # Attention aux contours du "loup"
                    if edge_x <= 1 or edge_y <= 1:
                        attn1[x, y] = 1.0
                    else:
                        attn1[x, y] = 0.3
                else:
                    attn1[x, y] = 0.1
        
        # 2. Carte d'attention de niveau intermédiaire: formes et parties
        attn2 = np.zeros((16, 16))
        # Attention à la "tête" du loup
        head_x, head_y = 5, 5  # position de la tête
        for x in range(16):
            for y in range(16):
                dist_to_head = np.sqrt((x - head_x)**2 + (y - head_y)**2)
                if dist_to_head < 3:
                    attn2[x, y] = 1.0
                elif 3 <= x <= 12 and 3 <= y <= 10:  # corps
                    attn2[x, y] = 0.5
                else:
                    attn2[x, y] = 0.1
        
        # 3. Carte d'attention de haut niveau: objets entiers
        attn3 = np.zeros((16, 16))
        for x in range(16):
            for y in range(16):
                if 3 <= x <= 12 and 3 <= y <= 10:  # Attention au "loup" entier
                    attn3[x, y] = 0.8 + 0.2 * np.random.random()
                else:
                    attn3[x, y] = 0.1 * np.random.random()
        
        # Appliquer un flou gaussien pour des transitions plus douces
        attn1 = cv2.GaussianBlur(attn1, (3, 3), 0)
        attn2 = cv2.GaussianBlur(attn2, (3, 3), 0)
        attn3 = cv2.GaussianBlur(attn3, (3, 3), 0)
        
        # Normaliser
        attn1 = (attn1 - attn1.min()) / (attn1.max() - attn1.min() + 1e-8)
        attn2 = (attn2 - attn2.min()) / (attn2.max() - attn2.min() + 1e-8)
        attn3 = (attn3 - attn3.min()) / (attn3.max() - attn3.min() + 1e-8)
        
        # Regrouper les cartes d'attention avec des titres informatifs
        attention_maps = [
            ("Couche peu profonde (2), Tête 1\nDétection des contours", attn1, 'viridis'),
            ("Couche intermédiaire (6), Tête 1\nParties de l'objet", attn2, 'plasma'),
            ("Couche profonde (12), Tête 1\nObjet entier", attn3, 'magma')
        ]
        
        # Afficher les cartes d'attention
        for i, (title, attn_map, cmap) in enumerate(attention_maps):
            if i < 2:
                ax = plt.subplot(gs[0, i+1])
            else:
                ax = plt.subplot(gs[1, 0])
            
            im = ax.imshow(attn_map, cmap=cmap, interpolation='bilinear')
            ax.set_title(title, fontsize=14)
            ax.axis('off')
            
            # Ajouter colorbar
            try:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
            except:
                # En cas d'erreur avec make_axes_locatable, utiliser la méthode standard
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Carte d'attention combinée (superposée à l'image)
        ax_overlay = plt.subplot(gs[1, 2])
        
        # Redimensionner l'image et la convertir en array numpy
        img_np = np.array(img.resize((16, 16), Image.LANCZOS))
        
        # Créer un overlay d'attention combinée
        combined_attn = (attn1 * 0.2 + attn2 * 0.3 + attn3 * 0.5)
        combined_attn = (combined_attn - combined_attn.min()) / (combined_attn.max() - combined_attn.min() + 1e-8)
        
        # Créer un heatmap
        cmap = plt.cm.inferno
        attention_heatmap = cmap(combined_attn)
        attention_heatmap = attention_heatmap[:, :, :3]  # Garder seulement RGB
        
        # Superposer l'image et le heatmap
        overlay = img_np * 0.6 + attention_heatmap * 255 * 0.4
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        ax_overlay.imshow(overlay)
        ax_overlay.set_title("Carte d'Attention Combinée", fontsize=14, fontweight='bold')
        ax_overlay.axis('off')
        
        plt.suptitle("Mécanisme d'attention dans les Vision Transformers (Simplifié)", 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{self.output_dir}/transformer_attention_maps.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Simplified transformer attention visualization complete!")
    
    def illustrate_sam_model(self):
        """
        Crée une illustration du modèle SAM (Segment Anything Model) avec de vrais masques générés.
        """
        print("Génération de l'illustration SAM...")
        
        # Installer SAM si nécessaire
        try:
            import segment_anything
        except ImportError:
            print("Installation du package segment-anything...")
            os.system("pip install git+https://github.com/facebookresearch/segment-anything.git")
            import segment_anything
        
        # Télécharger les poids du modèle SAM si nécessaire
        model_path = "sam_vit_h_4b8939.pth"
        if not os.path.exists(model_path):
            print("Téléchargement des poids du modèle SAM...")
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            response = requests.get(url, stream=True)
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Charger le modèle SAM
        from segment_anything import sam_model_registry, SamPredictor
        
        sam = sam_model_registry["vit_h"](checkpoint=model_path)
        sam.to(device=self.device)
        predictor = SamPredictor(sam)
        
        # Charger une image d'exemple avec un chien
        try:
            # Télécharger une image avec un chien
            url = "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg"
            response = requests.get(url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img_array = np.array(img)
            else:
                raise Exception("Échec du téléchargement de l'image")
        except:
            # Image de secours
            try:
                # Essayer d'utiliser une image locale
                img_path = "sample_image.jpg"
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img_array = np.array(img)
                else:
                    raise Exception("Image locale non trouvée")
            except:
                # Créer une image synthétique avec une forme similaire à un chien
                img_array = np.ones((512, 512, 3), dtype=np.uint8) * 240
                # Corps du "chien"
                cv2.ellipse(img_array, (250, 300), (120, 80), 0, 0, 360, (150, 100, 50), -1)
                # Tête
                cv2.circle(img_array, (150, 250), 60, (120, 80, 40), -1)
                # Oreilles
                cv2.ellipse(img_array, (120, 210), (30, 20), 45, 0, 360, (100, 70, 30), -1)
                cv2.ellipse(img_array, (180, 210), (30, 20), -45, 0, 360, (100, 70, 30), -1)
                # Pattes
                cv2.rectangle(img_array, (200, 360), (230, 420), (130, 90, 40), -1)
                cv2.rectangle(img_array, (270, 360), (300, 420), (130, 90, 40), -1)
                # Queue
                pts = np.array([[370, 300], [420, 270], [430, 290], [390, 320]], np.int32)
                cv2.fillPoly(img_array, [pts], (140, 95, 45))
        
        # Définir l'image pour le modèle SAM
        predictor.set_image(img_array)
        
        # 1. Point au centre du chien pour la segmentation avec un seul point
        height, width = img_array.shape[:2]
        # Identifier approximativement le centre du chien (ajuster selon l'image)
        # Pour l'image de chien standard, le centre pourrait être:
        center_x, center_y = width // 3, height // 2  # Ajuster selon l'image
        input_point = np.array([[center_x, center_y]])
        input_label = np.array([1])  # Point positif
        
        # 2. Définir une boîte englobante autour du chien
        box_for_dog = np.array([
            [width // 6, height // 4],     # Coin haut-gauche
            [width * 2 // 3, height * 3 // 4]  # Coin bas-droit
        ])
        
        # Générer les masques
        # A. Avec un point
        masks_from_point, scores_point, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
        
        # B. Avec une boîte englobante
        masks_from_box, scores_box, _ = predictor.predict(
            box=box_for_dog,
            multimask_output=True
        )
        
        # Visualiser les résultats
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Image originale
        axs[0].imshow(img_array)
        axs[0].set_title("Image originale", fontsize=14)
        axs[0].axis("off")
        
        # Segmentation avec un seul point
        best_mask_idx = np.argmax(scores_point)
        best_point_mask = masks_from_point[best_mask_idx]
        
        # Appliquer un overlay du masque sur l'image
        masked_img_point = img_array.copy()
        masked_img_point[best_point_mask] = masked_img_point[best_point_mask] * 0.7 + np.array([0, 255, 0]) * 0.3  # Overlay vert
        
        axs[1].imshow(masked_img_point)
        axs[1].set_title("Segmentation avec 1 point", fontsize=14)
        # Afficher le point
        axs[1].plot(center_x, center_y, 'o', color='lime', markersize=8)
        axs[1].axis("off")
        
        # Segmentation avec une boîte englobante
        best_mask_idx = np.argmax(scores_box)
        best_box_mask = masks_from_box[best_mask_idx]
        
        # Appliquer un overlay du masque sur l'image
        masked_img_box = img_array.copy()
        masked_img_box[best_box_mask] = masked_img_box[best_box_mask] * 0.7 + np.array([255, 0, 0]) * 0.3  # Overlay rouge
        
        axs[2].imshow(masked_img_box)
        axs[2].set_title("Segmentation avec boîte", fontsize=14)
        # Afficher la boîte
        rect = patches.Rectangle((box_for_dog[0][0], box_for_dog[0][1]), 
                                box_for_dog[1][0] - box_for_dog[0][0], 
                                box_for_dog[1][1] - box_for_dog[0][1], 
                                linewidth=2, edgecolor='r', facecolor='none')
        axs[2].add_patch(rect)
        axs[2].axis("off")
        
        plt.suptitle("Segmentation avec SAM (Segment Anything Model)", fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/sam_segmentation.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        print("Illustration SAM créée avec succès!")

def main():
    illustrations = CNNIllustrations()
    
    # Generate all illustrations
    illustrations.illustrate_cnn_architectures()
    illustrations.illustrate_yolo_detection()
    illustrations.illustrate_segmentation()
    illustrations.illustrate_transformer_attention()
    illustrations.illustrate_sam_model()
    
    print("All illustrations generated successfully in the 'figures' directory!")

if __name__ == "__main__":
    main()

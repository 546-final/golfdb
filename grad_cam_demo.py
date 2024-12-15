import argparse
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from eval import ToTensor, Normalize
from model import EventDetector
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
#from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class GradCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            GradCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        # 2D image
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))
        
        # 2D image (B, T, C, H, W)
        elif len(grads.shape) == 5:
            return np.mean(grads, axis=(1, 3, 4))
        
        else:
            raise ValueError("Invalid grads shape." 
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")
    
    def get_target_width_height(self, input_tensor: torch.Tensor):
        if len(input_tensor.shape) == 4:
            width, height = input_tensor.size(-1), input_tensor.size(-2)
            return width, height
        elif len(input_tensor.shape) == 5:
            width, height = input_tensor.size(-1), input_tensor.size(-2)
            return width, height
        else:
            raise ValueError("Invalid input_tensor shape. Only 2D or 3D images are supported.")

EVENT_NAMES = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing-1',
    3: 'Top',
    4: 'Mid-downswing-2',
    5: 'Impact',
    6: 'Mid-follow-through',
    7: 'Finish'
}


class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # preprocess and return frames
        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)

            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        labels = np.zeros(len(images)) # only for compatibility with transforms
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample

def create_labeled_image_grid(images, filename):
    """
    Create a grid of images with labels 0-8.
    
    Args:
    - images: List of 9 images (numpy arrays)
    
    Returns:
    - Grid image with labels
    """
    grid_size = 3
    h, w = images[0].shape[:2]
    grid_img = np.zeros((h*grid_size, w*grid_size, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        row = i // grid_size
        col = i % grid_size
        
        grid_img[row*h:(row+1)*h, col*w:(col+1)*w] = img
        
        cv2.putText(grid_img, 
                    str(i), 
                    (col*w+10, row*h+30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,255,0), 2)
    
    cv2.imwrite(filename+'.jpg', grid_img)
    return grid_img


def create_detailed_image_grid(images, 
                                row_labels=None, 
                                col_labels=None, 
                                figsize=(20, 15), 
                                spacing=0.1, 
                                title=None, 
                                filename=None):
    """
    Create a detailed image grid with row and column labels.
    
    Args:
    - images: 2D list of images (rows x columns)
    - row_labels: List of labels for rows
    - col_labels: Dictionary of column labels
    - figsize: Figure size (width, height) in inches
    - spacing: Spacing between subplots
    - title: Overall figure title
    - filename: Filename to save the figure
    
    Returns:
    - matplotlib figure object
    """
    # Determine grid dimensions
    rows = len(images)
    cols = len(images[0])
    
    # Create figure with specified size and spacing
    fig, axes = plt.subplots(rows, cols, figsize=figsize, 
                              gridspec_kw={'wspace': spacing, 'hspace': spacing})
    
    # Ensure axes is always 2D array
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Default column labels
    if col_labels is None:
        col_labels = EVENT_NAMES
    
    # Plot images
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            
            # Handle potentially missing images
            if i < len(images) and j < len(images[i]):
                img = images[i][j]
                
                # Handle different image formats
                if img.ndim == 2:  # Grayscale
                    ax.imshow(img, cmap='gray')
                elif img.shape[2] == 3:  # RGB
                    ax.imshow(img)
                elif img.shape[2] == 4:  # RGBA
                    ax.imshow(img)
                
                # Add column labels to top row
                if i == 0:
                    ax.set_title(col_labels.get(j, f'Column {j}'), fontsize=13, fontweight='bold')
                
                # Remove ticks
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Add row labels to first column
            if j == 0 and row_labels:
                ax.set_ylabel(row_labels[i], rotation=90, fontsize=13, fontweight='bold')
    
    # Add overall title if specified
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if filename provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_gradcam(model, dl, device, seq_length, args):
    """
    Perform inference and generate Grad-CAM visualizations for SwingNet model
    
    Args:
    - model: Trained SwingNet model
    - dl: DataLoader with input images
    - device: Torch device (cuda/cpu)
    - seq_length: Number of frames to process in each batch
    - args: Arguments containing video path
    """
    model.to(device)
    model.eval()
    
    # Identify target layers for Grad-CAM (depends on the model architecture)
    target_layers = [model.cnn[-1][-1]]  
    
    # Prepare Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    for sample in dl:
        images = sample['images']
        batch_probs = []
        batch_cams = []
        
        # Process images in batches due to memory constraints
        for batch in range(0, images.shape[1], seq_length):
            end_idx = min(batch + seq_length, images.shape[1])
            image_batch = images[:, batch:end_idx, :, :, :]
            
            # Flatten spatial dimensions for CAM
            B, T, C, H, W = image_batch.shape
            image_batch_flat = image_batch.view(B, T, C, H, W)
            
            # Generate per-frame targets (assuming multi-class classification)
            #targets = [ClassifierOutputTarget(torch.tensor([1]))  for i in range(image_batch_flat.shape[0])]
            # If targets is None, the highest scoring category (for every member in the batch) will be used.
            targets = None # if targets is None else [ClassifierOutputTarget(torch.tensor([1]))  for i in range(image_batch_flat.shape[0])]
            
            # Run inference and generate Grad-CAM
            #with torch.no_grad():
            logits = model(image_batch.to(device))
            probs = F.softmax(logits.data, dim=1).cpu().numpy()
            
            # Generate CAM for each frame
            model.train()
            image_batch_flat.require_grad = True
            grayscale_cams = cam(input_tensor=image_batch_flat.to(device), targets=targets)
            model.eval()
            
            batch_probs.append(probs)
            batch_cams.append(grayscale_cams)
        
        # Combine probabilities and CAMs
        probs = np.concatenate(batch_probs, axis=0)
        cams = np.concatenate(batch_cams, axis=0)
        
        # Visualize top predictions
        events = np.argmax(probs, axis=0)[:-1]
        confidences = [probs[e, i] for i, e in enumerate(events)]
        
        print('Predicted event frames:', events)
        print('Confidence:', [np.round(c, 3) for c in confidences])


        H, W = 354, 492
        # Visualization (1) video gen
        cap = cv2.VideoCapture(args.path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('gradcam_visualization.mp4', fourcc, 30.0, (H, W))
        
        for frame_idx in range(images.shape[1]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (H, W))
            
            # Check if current frame is an event frame
            if frame_idx in events:
                event_idx = list(events).index(frame_idx)
                cam_frame = cams[frame_idx]
                cam_resized = cv2.resize(cam_frame, (frame.shape[1], frame.shape[0]))
                visualization = show_cam_on_image(frame/255.0, cam_resized, use_rgb=True)
                
                # Add text for event stage and confidence
                text = f"Stage {event_idx+1}, Conf: {confidences[event_idx]:.2f}"
                cv2.putText(visualization, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
                out.write((visualization * 255).astype(np.uint8))
            else:
                cam_frame = cams[frame_idx]
                cam_resized = cv2.resize(cam_frame, (frame.shape[1], frame.shape[0]))
                visualization = show_cam_on_image(frame/255.0, cam_resized, use_rgb=True)
                out.write((visualization * 255).astype(np.uint8))

        
        # Optional: Visualize CAMs for each predicted frame
        # cap = cv2.VideoCapture(args.path)
        # for i, event_frame in enumerate(events):
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, event_frame)
        #     ret, frame = cap.read()
        #     if ret:
        #         # Resize CAM to match original image
        #         cam_resized = cv2.resize(cams[event_frame], (frame.shape[1], frame.shape[0]))
        #         visualization = show_cam_on_image(frame/255.0, cam_resized, use_rgb=True)
        #         cv2.imwrite(f'gradcam_frame_{event_frame}.jpg', visualization * 255)
        
        # cap.release()
        # break  # Process first batch

def get_cams(model, dl, cam, target, device, seq_length, args):
    """
    Perform inference and generate Grad-CAM visual
    """
    for sample in dl:
        images = sample['images']
        batch_probs = []
        batch_cams = []
        
        # Process images in batches due to memory constraints
        for batch in range(0, images.shape[1], seq_length):
            end_idx = min(batch + seq_length, images.shape[1])
            image_batch = images[:, batch:end_idx, :, :, :]
            
            # Flatten spatial dimensions for CAM
            B, T, C, H, W = image_batch.shape
            image_batch_flat = image_batch.view(B, T, C, H, W)
            
            targets = None if target is None else [ClassifierOutputTarget(target) for i in range(image_batch_flat.shape[0])]
            
            # Run inference and generate Grad-CAM
            #with torch.no_grad():
            logits = model(image_batch.to(device))
            probs = F.softmax(logits.data, dim=1).cpu().numpy()
            
            # Generate CAM for each frame
            model.train()
            image_batch_flat.require_grad = True
            grayscale_cams = cam(input_tensor=image_batch_flat.to(device), targets=targets)
            model.eval()
            
            batch_probs.append(probs)
            batch_cams.append(grayscale_cams)
        
        # Combine probabilities and CAMs
        probs = np.concatenate(batch_probs, axis=0)
        cams = np.concatenate(batch_cams, axis=0)

        return probs, cams

def visualize_gradcam_grid(model, dl, device, seq_length, args):
    """
    Perform inference and generate Grad-CAM visualizations for SwingNet model
    
    Args:
    - model: Trained SwingNet model
    - dl: DataLoader with input images
    - device: Torch device (cuda/cpu)
    - seq_length: Number of frames to process in each batch
    - args: Arguments containing video path
    """
    model.to(device)
    model.eval()
    
    # Identify target layers for Grad-CAM (depends on the model architecture)
    target_layers = [model.cnn[-4].conv[-1], model.cnn[-1][-1]]  
    
    # Prepare Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    probs, cams = get_cams(model, dl, cam, 
                            target=None,  
                            device=device, seq_length=seq_length, args=args)
    
    # Visualize top predictions
    events = np.argmax(probs, axis=0)[:-1]
    confidences = [probs[e, i] for i, e in enumerate(events)]
    
    print('Predicted event frames:', events)
    print('Confidence:', [np.round(c, 3) for c in confidences])

    cam_dict = {e:[] for e in events}

    for i in range(9):
        probs, cams = get_cams(model, dl, cam,
                                target=i,
                                device=device, seq_length=seq_length, args=args)
        
        for e in events:
            cam_dict[e].append(cams[e])

        
        # Optional: Visualize CAMs for each predicted frame
    cap = cv2.VideoCapture(args.path)
    for i, event_frame in enumerate(events):
        cap.set(cv2.CAP_PROP_POS_FRAMES, event_frame)
        ret, frame = cap.read()
        if ret:
            # Resize CAM to match original image
            visualizations = []
            for cam_img in cam_dict[event_frame]:
                cam_resized = cv2.resize(cam_img, (frame.shape[1], frame.shape[0]))
                visualization = show_cam_on_image(frame/255.0, cam_resized, use_rgb=True)
                visualizations.append(visualization)
            grid_img = create_labeled_image_grid(visualizations, f'gradcam_frame_{event_frame}')
    
    cap.release()


def get_gradcam_heatmap(model, dl, device, seq_length, args):
    """
    Perform inference and generate Grad-CAM visualizations for SwingNet model
    
    Args:
    - model: Trained SwingNet model
    - dl: DataLoader with input images
    - device: Torch device (cuda/cpu)
    - seq_length: Number of frames to process in each batch
    - args: Arguments containing video path
    """
    model.to(device)
    model.eval()
    
    # Identify target layers for Grad-CAM (depends on the model architecture)
    target_layers = [model.cnn[-2].conv[-1], model.cnn[-3].conv[-1], model.cnn[-1][-1]]  
    
    # Prepare Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)
    
    for sample in dl:
        images = sample['images']
        batch_probs = []
        batch_cams = []
        
        # Process images in batches due to memory constraints
        for batch in range(0, images.shape[1], seq_length):
            end_idx = min(batch + seq_length, images.shape[1])
            image_batch = images[:, batch:end_idx, :, :, :]
            
            # Flatten spatial dimensions for CAM
            B, T, C, H, W = image_batch.shape
            image_batch_flat = image_batch.view(B, T, C, H, W)
            
            targets = None 
            
            # Run inference and generate Grad-CAM
            logits = model(image_batch.to(device))
            probs = F.softmax(logits.data, dim=1).cpu().numpy()
            
            # Generate CAM for each frame
            model.train()
            grayscale_cams = cam(input_tensor=image_batch_flat.to(device), targets=targets)
            model.eval()
            
            batch_probs.append(probs)
            batch_cams.append(grayscale_cams)
        
        # Combine probabilities and CAMs
        probs = np.concatenate(batch_probs, axis=0)
        cams = np.concatenate(batch_cams, axis=0)
        
        # Visualize top predictions
        events = np.argmax(probs, axis=0)[:-1]
        confidences = [probs[e, i] for i, e in enumerate(events)]
        
        print('Predicted event frames:', events)
        print('Confidence:', [np.round(c, 3) for c in confidences])


        cap = cv2.VideoCapture(args.path)
        raw_images = []
        grad_cam_heatmaps = []
        for i, event_frame in enumerate(events):
            cap.set(cv2.CAP_PROP_POS_FRAMES, event_frame)
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_images.append(frame/255.0)
            if ret:
                # Resize CAM to match original image
                cam_resized = cv2.resize(cams[event_frame], (frame.shape[1], frame.shape[0]))
                visualization = show_cam_on_image(frame/255.0, cam_resized, use_rgb=True)

                grad_cam_heatmaps.append(visualization)
        

        create_detailed_image_grid([raw_images, grad_cam_heatmaps],
                                row_labels=['RGB', 'Grad-Cam'], 
                                col_labels=None, 
                                figsize=(20, 5), 
                                spacing=0.05, 
                                title=None, 
                                filename='test3.png')
        break  # Process first batch

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to video that you want to test', default='test_video.mp4')
    parser.add_argument('-s', '--seq-length', type=int, help='Number of frames to use per forward pass', default=64)
    args = parser.parse_args()
    seq_length = args.seq_length

    print('Preparing video: {}'.format(args.path))

    ds = SampleVideo(args.path, transform=transforms.Compose([ToTensor(),
                                Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])]))

    dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    try:
        save_dict = torch.load('models/swingnet_1800.pth.tar')
    except:
        print("Model weights not found. Download model weights and place in 'models' folder. See README for instructions")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model.load_state_dict(save_dict['model_state_dict'])

    #visualize_gradcam(model, dl, device, seq_length, args )
    #visualize_gradcam_grid(model, dl, device, seq_length, args)
    get_gradcam_heatmap(model, dl, device, seq_length, args)

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import sqlite3
from data import cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from skimage import transform as trans
from sklearn.preprocessing import normalize
from backbones import get_model
import argparse
from utils_config import get_config

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Model Loading Functions
def load_retinaface_model(model, pretrained_path, load_to_cpu):
    print(f'Loading RetinaFace model from {pretrained_path}')
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda())
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = {k.split('module.', 1)[-1] if k.startswith('module.') else k: v 
                           for k, v in pretrained_dict['state_dict'].items()}
    else:
        pretrained_dict = {k.split('module.', 1)[-1] if k.startswith('module.') else k: v 
                           for k, v in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict, strict=False)
    return model

# Load ArcFace model
parser = argparse.ArgumentParser(description="Distributed Arcface Training in Pytorch")
parser.add_argument("config", type=str, help="Path to the config file")

# Simulate command-line arguments (same as if you ran: python train_v2.py configs/ms1mv3_r50_onegpu)
custom_args = ["configs/ms1mv2_mbf"]  
args = parser.parse_args(custom_args)  # Pass the custom arguments list
cfg = get_config(args.config)

arcface = get_model(
    cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
checkpoint = torch.load("arcface_weights/model.pt", map_location="cuda")
arcface.load_state_dict(checkpoint)
arcface.eval()

### Initialize and Load Models
cfg = cfg_re50
retinaface = RetinaFace(cfg=cfg, phase='test')
retinaface = load_retinaface_model(retinaface, 'Resnet50_epoch_40.pth', load_to_cpu=False)
retinaface.eval()
retinaface.to(device)
cudnn.benchmark = True

# arcface = torch.load('arcface_model.pth', map_location=device)
# arcface.eval()
# arcface.to(device)

### Database Setup
def initialize_database():
    conn = sqlite3.connect('face_database.db')
    cursor = conn.cursor()
    return conn, cursor

conn, cursor = initialize_database()

### Embedding Class (Adapted from eval_ijbc.py)
class Embedding(object):
    def __init__(self, model, image_size=(112, 112)):
        self.model = model
        self.image_size = image_size
        self.src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        self.src[:, 0] += 8.0

    def get(self, rimg, landmarks):
        # Convert 10 landmarks (5 pairs) to 5-point format
        assert landmarks.shape == (10,), "Expected 10 landmarks (5 points x 2)"
        landmark5 = landmarks.reshape(5, 2)
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg, M, self.image_size, borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        img_flip = np.transpose(img_flip, (2, 0, 1))
        input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.float32)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data):
        imgs = torch.Tensor(batch_data).to(device)
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = self.model(imgs)
        feat = feat.reshape([batch_data.shape[0] // 2, 2 * feat.shape[1]])
        return feat.cpu().numpy()

embedding_handler = Embedding(arcface)

### Face Detection and Alignment
def detect_and_align(frame, retinaface):
    img = np.float32(frame)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
    scale_landms = torch.Tensor([im_width, im_height] * 5).to(device)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    with torch.no_grad():
        loc, conf, landms = retinaface(img)
    
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)
    boxes = decode(loc.data.squeeze(0), priors.data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), priors.data, cfg['variance'])
    landms = landms * scale_landms
    landms = landms.cpu().numpy()

    confidence_threshold = 0.6
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    nms_threshold = 0.4
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    boxes = boxes[keep]
    landms = landms[keep]
    scores = scores[keep]

    aligned_faces = []
    for i in range(len(boxes)):
        landmarks = landms[i]  # 10 values (5 points)
        input_blob = embedding_handler.get(frame, landmarks)
        aligned_faces.append(input_blob)
    
    return aligned_faces, boxes, scores

### Face Recognition
def recognize_face(frame, retinaface, arcface, cursor):
    aligned_faces, boxes, scores = detect_and_align(frame, retinaface)
    
    if len(aligned_faces) == 0:
        return []  # No faces detected
    
    # Extract embeddings
    embeddings = []
    for aligned_face in aligned_faces:
        feat = embedding_handler.forward_db(aligned_face)
        embeddings.append(feat[0])  # Take the first (averaged) embedding
    embeddings = np.array(embeddings)
    embeddings = normalize(embeddings)  # L2 normalize

    # Query database
    cursor.execute("SELECT name, embedding FROM faces")
    rows = cursor.fetchall()
    if not rows:
        return [("Unknown", score, 0.0, box) for score, box in zip(scores, boxes)]
    
    db_names = []
    db_embeddings = []
    for row in rows:
        name = row[0]
        emb_bytes = row[1]
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        db_names.append(name)
        db_embeddings.append(emb)
    db_embeddings = np.array(db_embeddings)

    # Compute similarities
    threshold = 0.5
    results = []
    for embedding, box, score in zip(embeddings, boxes, scores):
        similarities = np.dot(db_embeddings, embedding)
        max_similarity = np.max(similarities)
        idx = np.argmax(similarities)
        name = db_names[idx] if max_similarity > threshold else "Unknown"
        results.append((name, score, max_similarity, box))
    
    return results

### Webcam Recognition
def recognize_from_webcam(retinaface, arcface, cursor, conn):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        results = recognize_face(frame, retinaface, arcface, cursor)
        
        for name, det_confidence, similarity, box in results:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f"{name} (Sim: {similarity:.2f})"
            cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        print(f"Number of faces detected: {len(results)}")
        cv2.imshow('Recognize Face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    recognize_from_webcam(retinaface, arcface, cursor, conn)
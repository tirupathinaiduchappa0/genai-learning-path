"""
===================================================================================
ROLE REFERENCE — COMPUTER VISION ENGINEER (Industrial / Physical AI)
===================================================================================

PURPOSE: This is a KEYWORD REFERENCE CARD, not a teaching lesson. It captures
everything from a real Computer Vision Engineer JD so that in the future, when
revising or exploring this career direction, I can instantly recall:
  - "This role exists, here's what it involves."
  - "Here are ALL the keywords / tech / concepts it demands."
  - "Here's what I could learn later (ML / CV / image-video-audio processing)."

CONTEXT (why I'm keeping this):
  A founder/co-founder wanted to talk to me — a GenAI engineer — even though
  this role is Computer Vision (image/video/industrial inspection), which I
  DON'T currently have. It's an exploratory conversation; the GenAI angle is
  the bridge (see SECTION 4). Storing keywords now so they're not lost.

  As of now I'm focused on Generative AI. In the FUTURE I may learn ML,
  Computer Vision, and image/video/audio processing with AI's help — this file
  is the map of what that would require.

SECTIONS:
  1. The Role in One Line + Experience Level
  2. ALL Keywords from the JD (verbatim-organized, by category)
  3. Adjacent Keywords the JD Did NOT List (completes the picture)
  4. The GenAI Bridge (why they want to talk to a GenAI person)
  5. Future Learning Map (if I ever pursue CV — the keyword roadmap)
  6. Quick Self-Assessment (what I have vs what this needs)
===================================================================================
"""

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — THE ROLE IN ONE LINE + EXPERIENCE LEVEL
# ═══════════════════════════════════════════════════════════════════════════════

SECTION_1 = """
ONE-LINER:
  "Build AI-powered vision systems that inspect products on a factory floor —
   cameras + deep learning models + edge hardware + PLC machine control."

  It's PHYSICAL AI (a.k.a. Industrial CV / Machine Vision), NOT web/cloud SaaS.
  The model runs next to a conveyor belt, looks at parts, and triggers a
  rejection arm or tower light when it sees a defect.

EXPERIENCE LEVEL:
  - 3+ years hands-on Computer Vision (design → develop → DEPLOY).
  - Must have shipped to PRODUCTION (not just notebooks / Kaggle).
  - Full lifecycle owner: model + app + hardware integration + deployment.

ROLE ALIASES (same job, different titles you may see later):
  - Machine Vision Engineer
  - Industrial Vision Engineer
  - Physical AI Engineer / Edge AI Engineer
  - Vision Systems Engineer
  - AI Engineer (Manufacturing / Inspection)

DOMAIN:  Manufacturing, industrial automation, quality control (QC),
         smart factory / Industry 4.0.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — ALL KEYWORDS FROM THE JD (organized by category, verbatim-based)
# ═══════════════════════════════════════════════════════════════════════════════

SECTION_2 = """
━━━ 2A. CORE CV CONCEPTS (from JD) ━━━
  image processing · video processing · deep learning · object detection ·
  image classification · model optimization · scalable production solutions

━━━ 2B. KEY RESPONSIBILITIES (keyword-extracted) ━━━
  • industrial inspection · automation · analytics
  • Windows / Python applications
  • multi-camera setups (multi-camera integration)
  • PLC connectivity
  • modular architectures
  • camera loading · frame processing · inspection workflows
  • deep learning models: YOLO · PaddleOCR · barcode/QR detection
  • object detection · text recognition (OCR)
  • vision pipeline optimization · real-time performance · edge devices
  • hardware collaboration: camera placement · lighting setups · industrial PC integration
  • edge computing deployment · reliable production operation
  • workflow documentation · version control
  • physical AI systems · factory floors

━━━ 2C. SKILLS & QUALIFICATIONS (keyword-extracted) ━━━
  LANGUAGES:        Python · C# · C++
  CV / DL LIBS:     OpenCV · PyTorch · TensorFlow · industrial vision libraries
  TECHNIQUES:       OCR · template matching · defect detection
  UI FRAMEWORKS:    PyQt5 · WPF (or similar desktop GUI)
  HARDWARE/DEPLOY:  industrial PCs · edge computing devices · production AI deployment
  VISION HARDWARE:  industrial cameras · optics · lighting techniques (inspection accuracy)
  PROTOCOLS:        PLC · Modbus · OPC-UA
  ACTUATORS:        rejection units · tower lights (stack lights)
  SOFT SKILLS:      debugging · problem solving · documentation

━━━ 2D. THE SPECIFIC NAMED TECH (memorize these — they're the differentiators) ━━━
  YOLO         → real-time object detection (You Only Look Once)
  PaddleOCR    → open-source OCR toolkit (text extraction from images)
  barcode/QR   → 1D/2D code reading (pyzbar, zxing, OpenCV)
  OpenCV       → the CV workhorse library (image ops, filtering, contours)
  PyQt5 / WPF  → desktop app UI (operator-facing HMI screens)
  PLC/Modbus/OPC-UA → the language machines speak on a factory floor
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ADJACENT KEYWORDS THE JD DID NOT LIST (completes the picture)
# ═══════════════════════════════════════════════════════════════════════════════

SECTION_3 = """
The JD is written by a practitioner, so it skips "obvious" foundations and some
modern tooling. These are the keywords that BELONG to this role but weren't
spelled out — capture them so the mental map is complete.

━━━ 3A. CV FUNDAMENTALS (assumed, not stated) ━━━
  image segmentation (semantic / instance) · edge detection (Canny/Sobel) ·
  contour detection · morphological ops (erode/dilate) · thresholding (Otsu) ·
  color spaces (RGB/HSV/grayscale) · histogram equalization · image filtering
  (Gaussian/median/blur) · feature detection (SIFT/SURF/ORB) · Hough transforms ·
  optical flow · background subtraction · ROI (region of interest)

━━━ 3B. DEEP LEARNING ARCHITECTURES FOR VISION ━━━
  CNN (convolutional neural network) · ResNet · EfficientNet · MobileNet
  (edge-friendly) · VGG · Vision Transformers (ViT) · U-Net (segmentation) ·
  Faster R-CNN · SSD · Detectron2 · anchor boxes · IoU · NMS (non-max suppression) ·
  mAP (mean average precision) · transfer learning · fine-tuning · backbone/head

━━━ 3C. YOLO ECOSYSTEM (the star of the JD, expanded) ━━━
  YOLOv5 · YOLOv8 · YOLOv11 · Ultralytics · YOLO-NAS · bounding boxes ·
  confidence threshold · class labels · custom dataset training · .pt weights

━━━ 3D. DATA & TRAINING PIPELINE (not mentioned but essential) ━━━
  data annotation / labeling · Roboflow · CVAT · LabelImg · Label Studio ·
  data augmentation (flip/rotate/crop/mosaic/brightness) · dataset splitting ·
  class imbalance · synthetic data · train/val/test · overfitting · epochs ·
  batch size · learning rate · loss curves

━━━ 3E. MODEL OPTIMIZATION & EDGE DEPLOYMENT (JD says "optimize", here's how) ━━━
  ONNX · TensorRT · OpenVINO (Intel) · quantization (INT8/FP16) · pruning ·
  model distillation · NVIDIA Jetson (Nano/Xavier/Orin) · Google Coral (TPU) ·
  Raspberry Pi · FPGA · inference latency · FPS (frames per second) · throughput ·
  batching · GPU/CPU/NPU · CUDA · cuDNN

━━━ 3F. INDUSTRIAL CAMERA / VISION HARDWARE (expands "industrial cameras, optics") ━━━
  GigE Vision · USB3 Vision · Camera Link · GenICam (standard) · area-scan vs
  line-scan cameras · global vs rolling shutter · frame grabber · machine vision
  lenses (focal length/aperture) · telecentric lens · backlight / ring light /
  dome light / bar light · strobe lighting · camera calibration · distortion
  correction · field of view (FOV) · depth of field · resolution/DPI ·
  Basler / Cognex / Keyence / FLIR / Allied Vision (vendor names)

━━━ 3G. FACTORY / OT INTEGRATION (expands PLC/Modbus/OPC-UA) ━━━
  SCADA · HMI · MES (manufacturing execution system) · digital I/O · GPIO ·
  trigger signals · encoders · conveyor belts · reject/eject mechanism ·
  Siemens / Allen-Bradley / Beckhoff PLCs · TCP/IP · serial (RS-232/485) ·
  Ethernet/IP · PROFINET · OT vs IT networks · deterministic/real-time

━━━ 3H. MLOps / SOFTWARE ENGINEERING (the "production" glue) ━━━
  Git · Docker · CI/CD · logging · monitoring · model versioning · data drift ·
  model retraining · A/B testing · unit tests · exception handling · threading /
  multiprocessing (frame capture vs inference) · queues · REST API · gRPC
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — THE GENAI BRIDGE (why they want to talk to a GenAI person)
# ═══════════════════════════════════════════════════════════════════════════════

SECTION_4 = """
The founder wants a GenAI person even for a CV role because the two worlds are
CONVERGING. These are the bridge keywords — the ones where MY current GenAI
skills actually plug into their vision problem.

━━━ 4A. MULTIMODAL / VISION-LANGUAGE MODELS (the overlap zone) ━━━
  VLM (Vision-Language Model) · multimodal LLM · GPT-4V / GPT-4o (vision) ·
  Claude vision · Gemini vision · LLaVA · Qwen-VL · CLIP · BLIP-2 · Florence-2 ·
  image captioning · visual question answering (VQA) · grounding · OCR-free
  document understanding

━━━ 4B. WHERE GENAI HELPS INDUSTRIAL CV (talking points for the founder chat) ━━━
  • Zero/few-shot defect description — VLM describes a defect in plain language
    instead of training a class-specific detector for every new defect type.
  • Synthetic data generation — diffusion models (Stable Diffusion) create rare
    defect images to balance datasets.
  • Auto-labeling — a big VLM pre-labels images, humans just verify (speeds
    annotation 10x).
  • Natural-language operator interface — "show me all rejected parts from the
    last hour" → LLM queries the inspection DB (RAG over logs).
  • Report generation — LLM writes the shift QC summary from raw detections.
  • Prompt-based inspection config — describe a new inspection rule in text.

━━━ 4C. THE HONEST FRAMING (what to actually say) ━━━
  "I'm strong in GenAI/LLM/RAG/agentic systems. I don't have hands-on industrial
   CV yet, but I understand the DL foundations (CNNs, object detection, model
   deployment) and I see exactly where multimodal GenAI plugs into vision
   inspection. I'd ramp on OpenCV/YOLO/edge fast, and I bring the GenAI layer
   that most pure-CV engineers don't have."

  → This is EXPLORATORY. The bridge is real; be honest about the gap.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FUTURE LEARNING MAP (if I ever pursue CV — the keyword roadmap)
# ═══════════════════════════════════════════════════════════════════════════════

SECTION_5 = """
If I ever decide to add CV to my skillset, this is the ORDER to learn it in.
Pure keyword roadmap — no explanations (learn each with AI's help when the time
comes).

STAGE 0 — Prereqs:
  Python (have it) · NumPy · matplotlib · basic linear algebra · basic ML

STAGE 1 — Classic CV (OpenCV):
  read/show/save images · color spaces · resize/crop/rotate · thresholding ·
  blurring · edge detection · contours · morphology · template matching ·
  video capture · webcam streams · drawing on frames

STAGE 2 — Deep Learning foundations:
  neural nets · backprop · CNN · pooling · activation functions · PyTorch basics ·
  training loop · datasets/dataloaders · transfer learning

STAGE 3 — Object Detection & OCR:
  YOLO (Ultralytics) · train custom YOLO · bounding boxes · mAP/IoU/NMS ·
  PaddleOCR · Tesseract · barcode/QR (pyzbar)

STAGE 4 — Data pipeline:
  annotation (Roboflow/CVAT) · augmentation · dataset curation · handling
  class imbalance

STAGE 5 — Edge & Optimization:
  ONNX export · TensorRT/OpenVINO · quantization · Jetson deployment · FPS tuning ·
  threading for capture-vs-inference

STAGE 6 — Industrial integration:
  industrial cameras (GigE/USB3) · lighting · PLC/Modbus/OPC-UA · PyQt5 desktop
  app · trigger/reject logic

STAGE 7 — GenAI x CV (my differentiator):
  VLMs · CLIP · GPT-4V · synthetic data (diffusion) · auto-labeling · RAG over
  inspection logs

RESOURCES (search these, don't trust a memorized URL):
  • OpenCV official docs / "OpenCV Python tutorials"
  • Ultralytics YOLO docs (docs.ultralytics.com)
  • PyImageSearch (Adrian Rosebrock) — classic CV blog
  • "PyTorch for Computer Vision" course search
  • NVIDIA Jetson "Hello AI World" (dusty-nv GitHub)
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — QUICK SELF-ASSESSMENT (what I have vs what this needs)
# ═══════════════════════════════════════════════════════════════════════════════

SECTION_6 = """
                       HAVE NOW?   GAP TO CLOSE
  Python                 ✅ Strong    —
  DL concepts (CNN etc.) 🟡 Theory    Hands-on training/deployment
  OpenCV                 ❌ No        Stage 1 above
  YOLO / object detect   ❌ No        Stage 3 above
  PaddleOCR / OCR        ❌ No        Stage 3 above
  PyQt5 / WPF desktop    ❌ No        Stage 6 (also new: desktop vs web)
  C# / C++               ❌ Python-only  Optional (Python path exists)
  Edge (Jetson/ONNX)     ❌ No        Stage 5 above
  Industrial cameras     ❌ No        On-the-job / Stage 6
  PLC / Modbus / OPC-UA  ❌ No        On-the-job / Stage 6
  GenAI / LLM / RAG      ✅ Strong    ← MY EDGE, the bridge (Section 4)

VERDICT:
  This role is ~70% new skills for me today. It is NOT a fit for a pure-CV
  hire, BUT the founder chat is worth it because:
    (a) the GenAI-x-CV bridge is genuinely valuable and rare,
    (b) it maps a whole career direction I can grow into later,
    (c) low pressure — "just a conversation."
  Be honest about the gap, lead with the GenAI bridge, stay curious.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN LESSONS (the 60-second recall card)
# ═══════════════════════════════════════════════════════════════════════════════

GOLDEN_LESSONS = """
  1. This is PHYSICAL / INDUSTRIAL AI — model runs on a factory floor, not the cloud.
  2. The named tech to remember: YOLO · PaddleOCR · OpenCV · PyQt5/WPF · PLC/Modbus/OPC-UA.
  3. Full-stack CV role = model + desktop app + camera/lighting hardware + machine control.
  4. "Optimize for edge" = ONNX/TensorRT + quantization + Jetson + FPS, not just accuracy.
  5. The JD skips fundamentals (CNN, annotation, augmentation) — they're assumed.
  6. GenAI bridge is REAL: VLMs, auto-labeling, synthetic data, RAG over logs.
  7. My honest pitch: strong GenAI + DL theory, will ramp CV fast, bring the GenAI layer.
  8. This file is a MAP, not a lesson — revisit it if/when I pursue CV later.
"""


if __name__ == "__main__":
    for _name, _body in [
        ("SECTION 1 — ROLE + EXPERIENCE", SECTION_1),
        ("SECTION 2 — JD KEYWORDS", SECTION_2),
        ("SECTION 3 — ADJACENT KEYWORDS", SECTION_3),
        ("SECTION 4 — GENAI BRIDGE", SECTION_4),
        ("SECTION 5 — FUTURE LEARNING MAP", SECTION_5),
        ("SECTION 6 — SELF-ASSESSMENT", SECTION_6),
        ("GOLDEN LESSONS", GOLDEN_LESSONS),
    ]:
        print("=" * 83)
        print(_name)
        print("=" * 83)
        print(_body)


# Role Overview

# We are looking for a Computer Vision Engineer with 3+ years of experience in designing, developing, and deploying AI-powered computer vision solutions. The ideal candidate should have hands-on expertise in image processing, deep learning, object detection, image classification, and model optimization, along with experience delivering scalable solutions in production environments.

# Key Responsibilities

 

# Design and implement computer vision solutions for industrial inspection, automation, and analytics.
# Develop and maintain Windows/Python applications integrating multi‑camera setups and PLC connectivity.
# Build modular architectures for camera loading, frame processing, and inspection workflows.
# Apply deep learning models (YOLO, PaddleOCR, barcode/QR detection) for object detection and text recognition.
# Optimize vision pipelines for real‑time performance on edge devices.
# Collaborate with hardware teams to design camera placement, lighting setups, and industrial PC integration.
# Deploy AI models on edge computing platforms and ensure reliable operation in production environments.
# Document workflows, maintain version control, and support deployment in physical AI systems on factory floors.
# Skills & Qualifications

# Strong programming skills in Python, C#, or C++.
# Hands‑on experience with OpenCV, PyTorch/TensorFlow, and industrial vision libraries.
# Knowledge of OCR, template matching, and defect detection techniques.
# Experience with application frameworks (PyQt5, WPF, or similar) for UI development.
# Familiarity with industrial PCs, edge computing devices, and deployment of AI models in production.
# Practical know‑how of industrial cameras, optics, and lighting techniques for inspection accuracy.
# Understanding of industrial protocols (PLC, Modbus, OPC‑UA) and integration with rejection units/tower lights.
# Strong debugging, problem solving, and documentation skills.
# Driver Drowsiness Detection System

## Abstract

This project focuses on enhancing safety in autonomous vehicle systems by implementing a **driver drowsiness detection system**. Using a combination of facial detection algorithms and Convolutional Neural Networks (CNNs), the system monitors real-time dashcam footage to detect eye closure and yawning, both indicators of drowsiness. The system triggers alerts to ensure the driver remains vigilant, enhancing safety during autonomous driving.

---

## Research Problem

Road accidents caused by drowsy driving contribute to over 1.35 million deaths annually (WHO 2022). Autonomous driving (AD) systems aim to reduce such risks, but human factors such as drowsiness still pose challenges, particularly during **take-over requests** (TORs), when the system hands back control to the driver. The objective of this research was to develop a **real-time drowsiness detection system** that alerts drivers during critical moments, preventing potential accidents.

---

## Methodology

### Tools & Technologies

- **Facial Detection Algorithm:** Harr-like features combined with CNN models were applied for detecting driver drowsiness.
- **Technologies Used:** **OpenCV** for video stream processing.
- **Data Collection:** The YawDD dataset (Abtahi et al. 2014) was used to train the CNN models to detect yawning and eye closures from video footage.
  
### Model Architecture
<p>
  <img src="Figures\ClosedEyeModel.png" style="width: 1080px">
</p>

- **Eye Closure Detection:** A CNN was trained using eye images to detect drowsiness based on prolonged eye closure (e.g., greater than 2 seconds).
- **Yawning Detection:** Another CNN model was trained to detect yawning from mouth movement, using **grayscale image input**.
- **Alert System:** Alerts were triggered when a driver exhibited prolonged drowsiness symptoms. Alerts were shown both visually on the UI (red borders) and through auditory cues.

---

## Research Results

The **driver drowsiness detection system** demonstrated the following key outcomes:
<p float="left">
    <img src="Figures\Normal.jpg" style="width: 400px">
    <img src="Figures\Yawn.jpg" style="width: 400px">
</p>

- **Improved Response Time:** The system effectively detected drowsiness, issuing alerts within 1 second of detecting prolonged eye closure.
- **Accuracy:** CNN models achieved **93% accuracy** in detecting closed eyes and **85% accuracy** in detecting yawning under controlled lighting conditions.

The combination of eye closure and yawning detection provided reliable drowsiness prediction in simulated and real-life driving scenarios. Further refinement of the system could improve its performance under more variable conditions (e.g., different lighting or head positions).

---

## Future Work

There is considerable room for further development, particularly in the following areas:

- **Enhanced Face Detection:** Improved algorithms are needed to support facial feature extraction from various angles, especially for head tilts or rotations.
- **Light Variability:** Developing better pre-processing techniques to enhance image clarity under different lighting conditions (e.g., night driving).
- **Real-World Testing:** Conducting real-world validation to test the system's effectiveness in uncontrolled environments.
- **Scalability & Optimization:** Further research should focus on optimizing the model to reduce computational load and improve real-time performance for in-vehicle deployment.

---

## Related Skills

- **Human Factors Research:** Applied principles to create a user-centric monitoring and alert system for driver safety.
- **Machine Learning:** Trained and fine-tuned CNN models using large datasets to classify drowsiness symptoms with python.
---

## Contact Information

- **Email:** [petewachi@outlook.com](mailto:petewachi@outlook.com)
- **LinkedIn:** [Wachirawit Umpaipant](https://www.linkedin.com/in/umpaipantw)

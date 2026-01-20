
<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# QCOS Dashboard

**Repository:** [https://github.com/alast27-cyber/QCOS-chips](https://github.com/alast27-cyber/QCOS-chips)

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1sPtQgiBSQWNAkUMS8yR_x1o0YF_NzBXE

## Quantum Bypass Installation Guide

If you encounter the **"12-Byte Corrupted File"** error or **"QPU Missing"** alert during standard installation, follow these steps to perform a **Ground-Up Rebuild** via the QCOS Virtual Quantum Machine (VQM).

1.  **Access Gateway**: Launch the application.
2.  **Reset Local State**: If you are already stuck in a "Partial Install" loop, go to the Login Screen and click the **[ ! ] RESET SIMULATION & RE-INSTALL** button at the bottom.
3.  **Authenticate**: Log in with `admin` / `qcos@123`.
4.  **Installer Intercept**: The system will detect the missing binary and intercept the boot sequence.
5.  **Bypass Protocol**: The **ChipsBrowserInstaller** will automatically:
    *   Scan local storage.
    *   Detect the 12-byte stub.
    *   Initiate the VQM Bypass.
    *   Compile the full 100MB+ Quantum Density payload.
6.  **Genesis Handshake**: Once compilation reaches 100%, the **Entangled Key State (EKS)** will sync.
7.  **Launch**: Click **"LAUNCH CHIPS BROWSER"** to enter the fully installed environment.

## Run Locally

**Prerequisites:**  Node.js

1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

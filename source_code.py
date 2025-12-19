import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import pyaudio
import numpy as np
import time
from ultralytics import YOLO
import datetime

# --- SYSTEM CONFIGURATION ---
MIC_INDEX = 1
CHUNK = 2048
CONFIDENCE_THRESHOLD = 0.5

# --- COLORS (Enhanced Military HUD Theme) ---
COLOR_BG = "#050505"
COLOR_PANEL = "#0a0a0a"
COLOR_TEXT_MAIN = "#00FF41"
COLOR_TEXT_WARN = "#FFA500"
COLOR_TEXT_ALERT = "#FF0000"
COLOR_HUD_LINES = "#008F11"
COLOR_ACCENT = "#00FFFF"     # Cyan for highlights

class NetraMilitarySystem:
    def __init__(self, root):
        self.root = root
        self.root.title("NETRA DEFENSE SYSTEM - MULTI-MODAL THREAT INTELLIGENCE")
        self.root.geometry("1400x800")
        self.root.configure(bg=COLOR_BG)

        # Variables
        self.audio_alert = ""
        self.audio_alert_type = ""   # For color coding
        self.alert_timer = 0
        self.running = True
        self.drone_counter = 0
        self.live_vol = 0
        self.live_freq = 0
        self.live_zcr = 0.0
        self.frame_count = 0

        # Threat counters
        self.total_threats = 0
        self.audio_threats = 0
        self.visual_threats = 0

        # Visual tracking
        self.tracked_persons = {}   # Store person positions for loitering detection
        self.last_visual_alert = time.time()
        
        # --- UI LAYOUT ---
        self.build_ui()

        # --- START AI ENGINES ---
        self.log("SYSTEM INITIALIZING...", "SYSTEM")
        self.log("Loading YOLO model...", "SYSTEM")
        self.model = YOLO('yolov8n.pt')
        self.log("YOLO model loaded", "SYSTEM")

        self.log("Initializing video capture...", "SYSTEM")
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.log("Camera active", "SYSTEM")
        else:
            self.log("Camera initialization failed", "ERROR")
            
        # Start Audio Thread
        self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        self.audio_thread.start()
        
        # Start Video Loop
        self.update_loop()

    def build_ui(self):
        """Enhanced UI with better layout and statistics"""

        # 1. Header with system status
        header = tk.Frame(self.root, bg=COLOR_PANEL, height=60)
        header.pack(fill=tk.X, pady=2)
        header.pack_propagate(False)

        tk.Label(header, text=" ⬢  NETRA MULTI-MODAL THREAT FUSION  ⬢ ",
                 bg=COLOR_PANEL, fg=COLOR_TEXT_MAIN,
                 font=("Consolas", 18, "bold")).pack(pady=5)

        status_frame = tk.Frame(header, bg=COLOR_PANEL)
        status_frame.pack()

        self.status_audio = tk.Label(status_frame, text="● AUDIO", bg=COLOR_PANEL,
                                     fg="#00FF00", font=("Consolas", 9))
        self.status_audio.pack(side=tk.LEFT, padx=10)

        self.status_visual = tk.Label(status_frame, text="● VISUAL", bg=COLOR_PANEL,
                                      fg="#00FF00", font=("Consolas", 9))
        self.status_visual.pack(side=tk.LEFT, padx=10)

        self.status_fusion = tk.Label(status_frame, text="● FUSION", bg=COLOR_PANEL,
                                      fg="#00FF00", font=("Consolas", 9))
        self.status_fusion.pack(side=tk.LEFT, padx=10)

        # 2. Main Area
        main_frame = tk.Frame(self.root, bg=COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left: Video Feed
        video_frame = tk.Frame(main_frame, bg=COLOR_PANEL, relief="ridge", bd=2)
        video_frame.pack(side=tk.LEFT, padx=5, pady=5)

        tk.Label(video_frame, text="[ VISUAL SENSOR FEED ]", bg=COLOR_PANEL,
                 fg=COLOR_TEXT_MAIN, font=("Consolas", 10, "bold")).pack(pady=5)

        self.video_label = tk.Label(video_frame, bg="black", borderwidth=2, relief="solid")
        self.video_label.pack(padx=5, pady=5)
        
        # Right: HUD Panel
        right_panel = tk.Frame(main_frame, bg=COLOR_BG)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # --- STATISTICS PANEL ---
        stats_panel = tk.Frame(right_panel, bg=COLOR_PANEL, relief="ridge", bd=2)
        stats_panel.pack(fill=tk.X, pady=5)

        tk.Label(stats_panel, text="[ THREAT STATISTICS ]", bg=COLOR_PANEL,
                 fg=COLOR_TEXT_MAIN, font=("Consolas", 11, "bold")).pack(pady=5)

        stats_grid = tk.Frame(stats_panel, bg=COLOR_PANEL)
        stats_grid.pack(pady=10)

        # Total threats
        tk.Label(stats_grid, text="TOTAL THREATS:", bg=COLOR_PANEL,
                 fg="white", font=("Consolas", 9)).grid(row=0, column=0, sticky="w", padx=10)
        self.stat_total = tk.Label(stats_grid, text="0", bg=COLOR_PANEL,
                                   fg=COLOR_TEXT_ALERT, font=("Consolas", 14, "bold"))
        self.stat_total.grid(row=0, column=1, sticky="e", padx=10)

        # Audio threats
        tk.Label(stats_grid, text="AUDIO DETECTIONS:", bg=COLOR_PANEL,
                 fg="white", font=("Consolas", 9)).grid(row=1, column=0, sticky="w", padx=10, pady=5)
        self.stat_audio = tk.Label(stats_grid, text="0", bg=COLOR_PANEL,
                                   fg=COLOR_TEXT_WARN, font=("Consolas", 12, "bold"))
        self.stat_audio.grid(row=1, column=1, sticky="e", padx=10, pady=5)

        # Visual threats
        tk.Label(stats_grid, text="VISUAL DETECTIONS:", bg=COLOR_PANEL,
                 fg="white", font=("Consolas", 9)).grid(row=2, column=0, sticky="w", padx=10)
        self.stat_visual = tk.Label(stats_grid, text="0", bg=COLOR_PANEL,
                                    fg=COLOR_ACCENT, font=("Consolas", 12, "bold"))
        self.stat_visual.grid(row=2, column=1, sticky="e", padx=10)
        
        # --- AUDIO SENSOR DATA ---
        audio_panel = tk.Frame(right_panel, bg=COLOR_PANEL, relief="ridge", bd=2)
        audio_panel.pack(fill=tk.X, pady=5)

        tk.Label(audio_panel, text="[ AUDIO SENSOR DATA ]", bg=COLOR_PANEL,
                 fg=COLOR_TEXT_MAIN, font=("Consolas", 11, "bold")).pack(pady=8)

        # Volume Bar
        vol_frame = tk.Frame(audio_panel, bg=COLOR_PANEL)
        vol_frame.pack(pady=5)
        tk.Label(vol_frame, text="AMPLITUDE:", bg=COLOR_PANEL, fg="white",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=5)
        self.vol_bar = ttk.Progressbar(vol_frame, orient="horizontal",
                                       length=200, mode="determinate")
        self.vol_bar.pack(side=tk.LEFT, padx=5)
        self.vol_text = tk.Label(vol_frame, text="0", bg=COLOR_PANEL,
                                 fg=COLOR_TEXT_WARN, font=("Consolas", 10, "bold"))
        self.vol_text.pack(side=tk.LEFT, padx=5)
        
        # Frequency Display
        freq_frame = tk.Frame(audio_panel, bg=COLOR_PANEL)
        freq_frame.pack(pady=5)
        tk.Label(freq_frame, text="FREQUENCY:", bg=COLOR_PANEL, fg="white",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=5)
        self.freq_text = tk.Label(freq_frame, text="0 Hz", bg=COLOR_PANEL,
                                  fg=COLOR_ACCENT, font=("Consolas", 16, "bold"))
        self.freq_text.pack(side=tk.LEFT, padx=10)

        # ZCR Display (Roughness)
        zcr_frame = tk.Frame(audio_panel, bg=COLOR_PANEL)
        zcr_frame.pack(pady=5)
        tk.Label(zcr_frame, text="ROUGHNESS:", bg=COLOR_PANEL, fg="white",
                 font=("Consolas", 9)).pack(side=tk.LEFT, padx=5)
        self.zcr_text = tk.Label(zcr_frame, text="0.000", bg=COLOR_PANEL,
                                 fg="white", font=("Consolas", 12))
        self.zcr_text.pack(side=tk.LEFT, padx=10)

        tk.Label(audio_panel, text="─" * 40, bg=COLOR_PANEL,
                 fg=COLOR_HUD_LINES, font=("Consolas", 8)).pack(pady=5)
        
        # --- THREAT LOG ---
        log_panel = tk.Frame(right_panel, bg=COLOR_PANEL, relief="ridge", bd=2)
        log_panel.pack(fill=tk.BOTH, expand=True, pady=5)

        tk.Label(log_panel, text="[ THREAT INTELLIGENCE LOG ]", bg=COLOR_PANEL,
                 fg=COLOR_TEXT_MAIN, font=("Consolas", 11, "bold")).pack(pady=8)

        log_scroll_frame = tk.Frame(log_panel, bg=COLOR_PANEL)
        log_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = tk.Scrollbar(log_scroll_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_box = tk.Text(log_scroll_frame, height=15, bg="#000000",
                               fg=COLOR_TEXT_MAIN, font=("Consolas", 9),
                               state="disabled", yscrollcommand=scrollbar.set)
        self.log_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_box.yview)

        # Configure tags for colored text
        self.log_box.tag_config("CRITICAL", foreground="#FF0000")
        self.log_box.tag_config("DANGER", foreground="#FF4500")
        self.log_box.tag_config("HIGH", foreground="#FFA500")
        self.log_box.tag_config("AERIAL", foreground="#FF00FF")
        self.log_box.tag_config("VISUAL", foreground="#00FFFF")
        self.log_box.tag_config("SYSTEM", foreground="#00FF41")
        self.log_box.tag_config("ERROR", foreground="#FF0000")

    def log(self, msg, level="INFO"):
        """Enhanced logging with color coding"""
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.config(state="normal")

        if level in ["CRITICAL", "DANGER", "HIGH", "AERIAL", "VISUAL", "SYSTEM", "ERROR"]:
            self.log_box.insert(tk.END, f"[{ts}] ", "SYSTEM")
            self.log_box.insert(tk.END, f"{msg}\n", level)
        else:
            self.log_box.insert(tk.END, f"[{ts}] {msg}\n")

        self.log_box.see(tk.END)
        self.log_box.config(state="disabled")

    def audio_loop(self):
        """
        FIXED: Accurate audio detection based on your test data
        """
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100,
                            input=True, input_device_index=MIC_INDEX,
                            frames_per_buffer=CHUNK)
            self.log(f"Audio sensor active (MIC INDEX {MIC_INDEX})", "SYSTEM")
        except Exception as e:
            self.log(f"Microphone error: {e}", "ERROR")
            return
            
        consecutive_drone_frames = 0

        while self.running:
            try:
                # Read raw data
                raw = stream.read(CHUNK, exception_on_overflow=False)
                data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)

                # 1. Volume (RMS) - MAX VOLUME from your tests
                vol = np.sqrt(np.mean(data**2))

                # 2. Frequency (FFT) - AVG FREQUENCY from your tests
                fft_data = np.fft.rfft(data)
                fft_freqs = np.fft.rfftfreq(len(data), 1.0/44100)
                fft_mag = np.abs(fft_data)

                # Get dominant frequency
                peak_idx = np.argmax(fft_mag)
                freq = fft_freqs[peak_idx]

                # 3. Zero Crossing Rate (Roughness) - ROUGHNESS from your tests
                zero_crossings = np.where(np.diff(np.sign(data)))[0]
                zcr = len(zero_crossings) / len(data)

                # Update UI Variables
                self.live_vol = int(vol)
                self.live_freq = int(freq)
                self.live_zcr = zcr
                
                # --- THREAT DETECTION LOGIC (Based on YOUR test data) ---

                # 1. GUNSHOT DETECTION
                # Test data: MAX VOLUME: 4968, AVG FREQUENCY: 3438 Hz, ROUGHNESS: 0.067
                if vol > 3000:   # Lower threshold for sensitivity
                    if 3000 < freq < 3800 and zcr < 0.10:
                        self.trigger_alert("GUNSHOT DETECTED", "CRITICAL")
                        self.audio_threats += 1
                        self.total_threats += 1
                        time.sleep(1)
                        continue
                        
                # 2. BOMB/EXPLOSION DETECTION
                # Test data: MAX VOLUME: 6627, AVG FREQUENCY: 1848 Hz, ROUGHNESS: 0.065
                if vol > 4000:   # High volume
                    if 1500 < freq < 2100 and zcr < 0.10:
                        self.trigger_alert("EXPLOSION DETECTED", "DANGER")
                        self.audio_threats += 1
                        self.total_threats += 1
                        time.sleep(1)
                        continue
                        
                # 3. HUMAN SCREAM DETECTION
                # Test data: MAX VOLUME: 10257, AVG FREQUENCY: 2059 Hz, ROUGHNESS: 0.051
                if vol > 6000:   # Very high volume
                    if 1900 < freq < 2400 and zcr < 0.08:
                        self.trigger_alert("HUMAN DISTRESS CALL", "HIGH")
                        self.audio_threats += 1
                        self.total_threats += 1
                        time.sleep(1)
                        continue
                        
                # 4. DRONE DETECTION (Most sensitive - requires consistency)
                # Test data: MAX VOLUME: 335, AVG FREQUENCY: 7007 Hz, ROUGHNESS: 0.270
                if 6000 < freq < 8000 and zcr > 0.15:   # High frequency + high roughness
                    consecutive_drone_frames += 1
                    if consecutive_drone_frames > 8:   # Need 8 consecutive frames
                        self.trigger_alert("UAV/DRONE DETECTED", "AERIAL")
                        self.audio_threats += 1
                        self.total_threats += 1
                        consecutive_drone_frames = 0   # Reset
                        time.sleep(1)
                else:
                    # Reset if conditions not met
                    if consecutive_drone_frames > 0:
                        consecutive_drone_frames -= 1
                        
            except Exception as e:
                print(f"Audio error: {e}")
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

    def trigger_alert(self, name, alert_type):
        """Trigger audio alert with type for color coding"""
        self.audio_alert = name
        self.audio_alert_type = alert_type
        self.alert_timer = 40   # Show for 40 frames (~1.3 seconds)
        self.log(f"!!! {name} !!!", alert_type)

    def update_loop(self):
        """Enhanced video loop with better threat detection"""

        # 1. Update Sensor HUD
        self.vol_text.config(text=f"{self.live_vol}")
        self.freq_text.config(text=f"{self.live_freq} Hz")
        self.zcr_text.config(text=f"{self.live_zcr:.3f}")

        # Update statistics
        self.stat_total.config(text=str(self.total_threats))
        self.stat_audio.config(text=str(self.audio_threats))
        self.stat_visual.config(text=str(self.visual_threats))

        # Volume bar (scaled for visibility)
        vol_scaled = min(100, (self.live_vol / 100))
        self.vol_bar['value'] = vol_scaled

        # Change volume bar color based on level
        if self.live_vol > 5000:
            self.vol_bar.configure(style="red.Horizontal.TProgressbar")
        elif self.live_vol > 2000:
            self.vol_bar.configure(style="yellow.Horizontal.TProgressbar")
            
        # 2. Vision System
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (900, 650))
            frame_height, frame_width = frame.shape[:2]

            current_time = time.time()
            persons_in_frame = []

            # Run YOLO detection
            results = self.model(frame, stream=True, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.model.names[cls]
                    
                    if conf > CONFIDENCE_THRESHOLD:
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # --- PERSON DETECTION (Class 0) ---
                        if cls == 0:
                            persons_in_frame.append((center_x, center_y))
                            
                            # Draw red tactical box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                            # Military corners
                            corner_len = 25
                            cv2.line(frame, (x1, y1), (x1+corner_len, y1), (0,0,255), 5)
                            cv2.line(frame, (x1, y1), (x1, y1+corner_len), (0,0,255), 5)
                            cv2.line(frame, (x2, y1), (x2-corner_len, y1), (0,0,255), 5)
                            cv2.line(frame, (x2, y1), (x2, y1+corner_len), (0,0,255), 5)
                            cv2.line(frame, (x1, y2), (x1+corner_len, y2), (0,0,255), 5)
                            cv2.line(frame, (x1, y2), (x1, y2-corner_len), (0,0,255), 5)
                            cv2.line(frame, (x2, y2), (x2-corner_len, y2), (0,0,255), 5)
                            cv2.line(frame, (x2, y2), (x2, y2-corner_len), (0,0,255), 5)

                            # Label
                            label = f"INTRUDER [{int(conf*100)}%]"
                            cv2.putText(frame, label, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                            # Log detection (throttled)
                            if current_time - self.last_visual_alert > 3:
                                self.log(f"Intruder detected - Confidence: {conf:.2f}", "VISUAL")
                                self.visual_threats += 1
                                self.total_threats += 1
                                self.last_visual_alert = current_time

                        # --- DRONE/AERIAL DETECTION (Bird: 14, Airplane: 4, Kite: 37) ---
                        elif cls in [4, 14, 37]:
                            # Draw purple tactical box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                            # Corners
                            corner_len = 25
                            cv2.line(frame, (x1, y1), (x1+corner_len, y1), (255,0,255), 5)
                            cv2.line(frame, (x1, y1), (x1, y1+corner_len), (255,0,255), 5)
                            cv2.line(frame, (x2, y1), (x2-corner_len, y1), (255,0,255), 5)
                            cv2.line(frame, (x2, y1), (x2, y1+corner_len), (255,0,255), 5)

                            # Label
                            label = f"AERIAL THREAT [{int(conf*100)}%]"
                            cv2.putText(frame, label, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                            # Target marker
                            cv2.circle(frame, (center_x, center_y), 8, (255, 0, 255), 2)
                            cv2.circle(frame, (center_x, center_y), 15, (255, 0, 255), 1)

                            # Log detection (throttled)
                            if current_time - self.last_visual_alert > 3:
                                self.log(f"Aerial threat detected ({class_name}) - Confidence: {conf:.2f}", "AERIAL")
                                self.visual_threats += 1
                                self.total_threats += 1
                                self.last_visual_alert = current_time

                        # --- VEHICLE DETECTION (Car: 2, Truck: 7, Bus: 5, Motorcycle: 3) ---
                        elif cls in [2, 3, 5, 7]:
                            # Draw yellow box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            label = f"VEHICLE: {class_name.upper()} [{int(conf*100)}%]"
                            cv2.putText(frame, label, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        # --- BACKPACK/BAG DETECTION (Backpack: 24, Handbag: 26, Suitcase: 28) ---
                        elif cls in [24, 26, 28]:
                            # Draw orange box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                            label = f"SUSPICIOUS ITEM: {class_name.upper()}"
                            cv2.putText(frame, label, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            # 3. Draw Enhanced HUD Overlay
            # Crosshair (center)
            cx, cy = frame_width // 2, frame_height // 2
            cv2.line(frame, (cx-30, cy), (cx+30, cy), (0, 255, 0), 2)
            cv2.line(frame, (cx, cy-30), (cx, cy+30), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 40, (0, 255, 0), 2)

            # Corner brackets (full frame)
            bracket_len = 50
            # Top-left
            cv2.line(frame, (10, 10), (10+bracket_len, 10), (0, 255, 65), 3)
            cv2.line(frame, (10, 10), (10, 10+bracket_len), (0, 255, 65), 3)
            # Top-right
            cv2.line(frame, (frame_width-10, 10), (frame_width-10-bracket_len, 10), (0, 255, 65), 3)
            cv2.line(frame, (frame_width-10, 10), (frame_width-10, 10+bracket_len), (0, 255, 65), 3)
            # Bottom-left
            cv2.line(frame, (10, frame_height-10), (10+bracket_len, frame_height-10), (0, 255, 65), 3)
            cv2.line(frame, (10, frame_height-10), (10, frame_height-10-bracket_len), (0, 255, 65), 3)
            # Bottom-right
            cv2.line(frame, (frame_width-10, frame_height-10), (frame_width-10-bracket_len, frame_height-10), (0, 255, 65), 3)
            cv2.line(frame, (frame_width-10, frame_height-10), (frame_width-10, frame_height-10-bracket_len), (0, 255, 65), 3)

            # System info overlay (top-left)
            info_text = [
                f"NETRA v2.0 | {datetime.datetime.now().strftime('%H:%M:%S')}",
                f"FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))} | RES: {frame_width}x{frame_height}",
            ]
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (15, 25 + i*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 65), 1)
            
            # 4. Audio Alert Banner (with color coding)
            if self.alert_timer > 0:
                self.alert_timer -= 1

                # Choose color based on alert type
                if self.audio_alert_type == "CRITICAL":
                    banner_color = (0, 0, 255)   # Red
                elif self.audio_alert_type == "DANGER":
                    banner_color = (0, 69, 255)   # Orange-Red
                elif self.audio_alert_type == "HIGH":
                    banner_color = (0, 165, 255)   # Orange
                elif self.audio_alert_type == "AERIAL":
                    banner_color = (255, 0, 255)   # Purple
                else:
                    banner_color = (0, 255, 255)   # Yellow

                # Pulsing effect
                alpha = 0.7 if self.alert_timer % 10 < 5 else 0.9
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame_width, 80), banner_color, -1)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Alert text
                text_lines = [
                    " ⚠  THREAT ALERT  ⚠ ",
                    self.audio_alert
                ]
                for i, text in enumerate(text_lines):
                    font_scale = 1.2 if i == 0 else 0.9
                    thickness = 3 if i == 0 else 2
                    cv2.putText(frame, text, (30, 30 + i*35),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                                
            # 5. Render to UI
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)
            self.root.after(30, self.update_loop)

    def on_close(self):
        """Clean shutdown"""
        self.log("System shutting down...", "SYSTEM")
        self.running = False
        time.sleep(0.5)
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    # Configure ttk styles for progress bar colors
    style = ttk.Style()
    style.theme_use('default')
    style.configure("TProgressbar", thickness=20)
    
    app = NetraMilitarySystem(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

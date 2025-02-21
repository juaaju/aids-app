# ui/components/system_monitor.py
from flet import *
import psutil
import GPUtil
import threading
import time

# ui/components/system_monitor.py
from flet import *
import psutil
import GPUtil
import threading
import time

class SystemMonitor(Container):
    def __init__(self):
        super().__init__()
        self.bgcolor = colors.WHITE
        self.border_radius = border_radius.all(10)
        self.padding = padding.all(15)  # Sesuaikan padding
        self.margin = margin.only(bottom=20)
        self.width = 800  # Tambahkan fixed width
        
        # Initialize metrics dengan font size lebih kecil
        self.cpu_text = Text("CPU: 0%", color=colors.BLACK, size=12)
        self.memory_text = Text("Memory: 0%", color=colors.BLACK, size=12)
        self.gpu_text = Text("GPU: N/A", color=colors.BLACK, size=12)
        self.disk_text = Text("Disk: 0%", color=colors.BLACK, size=12)

        # Create progress bars dengan width lebih kecil
        self.cpu_progress = ProgressBar(width=80, height=4, color=colors.BLUE)
        self.memory_progress = ProgressBar(width=80, height=4, color=colors.GREEN)
        self.gpu_progress = ProgressBar(width=80, height=4, color=colors.RED)
        self.disk_progress = ProgressBar(width=80, height=4, color=colors.YELLOW)

        # Layout setup dengan spacing lebih kecil
        self.content = Row(
            controls=[
                Column(
                    controls=[self.cpu_text, self.cpu_progress],
                    spacing=2,
                    horizontal_alignment=CrossAxisAlignment.CENTER
                ),
                Container(width=1, bgcolor=colors.WHITE60, height=30),  # Divider
                Column(
                    controls=[self.memory_text, self.memory_progress],
                    spacing=2,
                    horizontal_alignment=CrossAxisAlignment.CENTER
                ),
                Container(width=1, bgcolor=colors.WHITE60, height=30),  # Divider
                Column(
                    controls=[self.gpu_text, self.gpu_progress],
                    spacing=2,
                    horizontal_alignment=CrossAxisAlignment.CENTER
                ),
                Container(width=1, bgcolor=colors.WHITE60, height=30),  # Divider
                Column(
                    controls=[self.disk_text, self.disk_progress],
                    spacing=2,
                    horizontal_alignment=CrossAxisAlignment.CENTER
                )
            ],
            alignment=MainAxisAlignment.CENTER,
            spacing=10  # Kurangi spacing antar komponen
        )

        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.update_metrics, daemon=True)
        self.monitor_thread.start()

    def update_metrics(self):
        while self.monitoring:
            try:
                # CPU Usage
                cpu_percent = psutil.cpu_percent()
                self.cpu_text.value = f"CPU: {cpu_percent}%"
                self.cpu_progress.value = cpu_percent / 100

                # Memory Usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.memory_text.value = f"Memory: {memory_percent}%"
                self.memory_progress.value = memory_percent / 100

                # GPU Usage (if available)
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        self.gpu_text.value = f"GPU: {gpu.load*100:.1f}%"
                        self.gpu_progress.value = gpu.load
                    else:
                        self.gpu_text.value = "GPU: N/A"
                        self.gpu_progress.value = 0
                except:
                    self.gpu_text.value = "GPU: N/A"
                    self.gpu_progress.value = 0

                # Disk Usage
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                self.disk_text.value = f"Disk: {disk_percent}%"
                self.disk_progress.value = disk_percent / 100

                # Update UI
                self.update()
                time.sleep(1)  # Update every second

            except Exception as e:
                print(f"Error updating metrics: {e}")
                time.sleep(1)

    def cleanup(self):
        self.monitoring = False
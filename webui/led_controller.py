import time
import threading

import adafruit_pixelbuf
import board
from adafruit_raspberry_pi5_neopixel_write import neopixel_write


class _Pi5Pixelbuf(adafruit_pixelbuf.PixelBuf):
    """PixelBuf that transmits via the Pi 5's RP1 PIO (any GPIO pin)."""

    def __init__(self, pin, size, **kwargs):
        self._pin = pin
        super().__init__(size=size, **kwargs)

    def _transmit(self, buf):
        neopixel_write(self._pin, buf)


class LEDController:
    """
    LED controller
    """
    # Defaults
    LED_PIN = 13
    LED_BRIGHTNESS = 64
    LED_COUNT = 18

    # LED indices
    RING_START_INDEX = 0
    RING_END_INDEX = 17

    def __init__(
        self,
        led_pin: int = LED_PIN,
        led_brightness: int = LED_BRIGHTNESS,
    ):
        """
        Initialize LED controller.

        Args:
            led_pin: GPIO pin for LED data (BCM number)
            led_brightness: Brightness (0-255)
        """

        self._animation_thread = None
        self._animation_stop = threading.Event()
        self.strip = _Pi5Pixelbuf(
                getattr(board, f"D{led_pin}"),
                self.LED_COUNT,
                auto_write=False,
                byteorder="GRB",
                brightness=led_brightness / 255,
            )
        self.all_off()
    
    # --- Color definitions ---
    
    @staticmethod
    def color_off():
        return (0, 0, 0)
    
    @staticmethod
    def color_red():
        return (255, 0, 0)

    @staticmethod
    def color_green():
        return (0, 255, 0)
    
    @staticmethod
    def color_blue():
        return (0, 0, 255)

    @staticmethod
    def color_yellow():
        return (255, 255, 0)
    
    @staticmethod
    def color_rgb(r: int, g: int, b: int):
        """Custom RGB color."""
        return (r, g, b)

    @staticmethod
    def dim_color(color, factor: float):
        """Dim a color by a factor (0.0 to 1.0)."""
        (r, g, b) = color
        return (int(r * factor), int(g * factor), int(b * factor))
    
    # --- Basic control ---
    
    def set_pixel(self, index: int, color):
        """Set a single led to a color."""
        if color is None:
            return

        if 0 <= index < self.LED_COUNT:
            (r, g, b) = color

            # The ring LEDs are RGB, while the others are GRB
            color_output = (g, r, b) if index < self.RING_START_INDEX else (r, g, b)

            self.strip[index] = color_output
    
    def all_off(self):
        """Turn all LEDs off."""
        for i in range(self.LED_COUNT):
            self.set_pixel(i, self.color_off())

        self.strip.show()
    
    def all_on(self, color):
        """Turn all LEDs to a specific color."""
        for i in range(self.LED_COUNT):
            self.set_pixel(i, color)

        self.strip.show()

    # --- Ring control ---
    
    def ring_on(self, color):
        """Turn on the LED ring (LEDs 4-15) with a specific color."""
        for i in range(self.RING_START_INDEX, self.RING_END_INDEX + 1):
            self.set_pixel(i, color)
        
        self.strip.show()
    
    def ring_off(self):
        """Turn off the LED ring."""
        self.ring_on(self.color_off())
    
    # --- Animations ---
    
    def stop_animation(self):
        """Stop any running animation."""
        self._animation_stop.set()

        if self._animation_thread and self._animation_thread.is_alive():
            self._animation_thread.join(timeout=2.0)

        self._animation_stop.clear()
    
    def pulse_ring(self, color, duration: float = 3600.0, step_time: float = 0.04):
        """
        Pulse the ring LEDs.
        
        Args:
            color: Base color to pulse
            duration: How long to pulse (seconds)
            step_time: Pulse speed (seconds per step)
        """
        self.stop_animation()
        
        def _pulse():
            start_time = time.time()

            while not self._animation_stop.is_set() and (time.time() - start_time) < duration:
                # Fade in
                for brightness in range(0, 256, 4):
                    if self._animation_stop.is_set():
                        break
                    self.ring_on(self.dim_color(color, brightness / 255))
                    time.sleep(step_time)
                
                # Fade out
                for brightness in range(255, -1, -4):
                    if self._animation_stop.is_set():
                        break
                    self.ring_on(self.dim_color(color, brightness / 255))
                    time.sleep(step_time)

            self.ring_off()
        
        self._animation_thread = threading.Thread(target=_pulse, daemon=False)
        self._animation_thread.start()
    
    def spin_ring(self, color, duration: float = 3600.0, step_time: float = 0.1, amount: int = 3):
        """
        Spinning animation on the ring.
        
        Args:
            color: Color for the spinning dot
            duration: How long to spin (seconds)
            step_time: Speed of rotation (seconds per step)
            amount: Number of LEDs lit at once
        """
        self.stop_animation()
        
        def _spin():
            start_time = time.time()

            while not self._animation_stop.is_set() and (time.time() - start_time) < duration:
                for i in range(self.RING_START_INDEX, self.RING_END_INDEX + 1):
                    if self._animation_stop.is_set():
                        break
                    self.set_pixel(i, color)
                    self.strip.show()

                    time.sleep(step_time)

                    off_index = ((i - amount - self.RING_START_INDEX + 1) % (self.RING_END_INDEX - self.RING_START_INDEX + 1)) + self.RING_START_INDEX
                    self.set_pixel(off_index, self.color_off())
                    self.strip.show()

            self.ring_off()
        
        self._animation_thread = threading.Thread(target=_spin, daemon=False)
        self._animation_thread.start()
    
    # --- Cleanup ---
    
    def cleanup(self):
        """Clean up LED resources."""
        self.stop_animation()
        self.all_off()
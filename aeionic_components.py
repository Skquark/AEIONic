
from collections import namedtuple
from typing import Optional, Callable
import flet as ft
import flet.canvas as cv
import os, re, time, base64, threading, cv2
from PIL import Image as PILImage

class SizeAwareControl(cv.Canvas):
    def __init__(self, content: Optional[ft.Control] = None, resize_interval: int=100, on_resize: Optional[Callable]=None, **kwargs):
        """
        :param content: A child Control contained by the SizeAwareControl. Defaults to None.
        :param resize_interval: The resize interval. Defaults to 100.
        :param on_resize: The callback function for resizing. Defaults to None.
        :param kwargs: Additional keyword arguments(see Canvas properties).
        Taken from: https://github.com/ndonkoHenri/Flet-Custom-Controls/commit/e1958e998ef0b16449cb58a13a94251f38ab2dac
        MIT license: https://github.com/ndonkoHenri/Flet-Custom-Controls/commit/8a8e920f0382734eef09734ea87a4c18d2cd21ed#diff-c693279643b8cd5d248172d9c22cb7cf4ed163a3c98c8a3f69c2717edd3eacb7
        """
        super().__init__(**kwargs)
        self.content = content
        self.resize_interval = resize_interval
        self.on_resize = self.__handle_canvas_resize
        self.resize_callback = on_resize
        self.size = namedtuple("size", ["width", "height"], defaults=[0, 0])

    def __handle_canvas_resize(self, e):
        """
        Called every resize_interval when the canvas is resized.
        If a resize_callback was given, it is called.
        """
        self.size = (int(e.width), int(e.height))
        self.update()
        if self.resize_callback:
            self.resize_callback(e)
#from size_aware_control import SizeAwareControl

"""Implements a pan and zoom control for flet UI framework."""
class PanZoom(ft.Row):
    """Pan and zoom control for flet UI framework.

    This control can be used to display a large image or other content that can be larger than the
    viewport.

    Important: the width and the height of the content must be specified in the constructor, since
    it is a write-only property. Use PIL (pillow) to figure out the width and height of an image.

    By default, the content is centered in the viewport, and scaled to fit in the viewport with
    padding added to the sides or top and bottom if necessary.

    No warranty, no support, use at your own risk, etc.
    """
    content_with_padding: ft.Container or None

    def __init__(self, content: ft.Control, content_width: int, content_height: int,
                 width: int = None, height: int = None, padding_color=ft.Colors.TRANSPARENT,
                 on_pan_update=None, on_scroll=None, on_click=None, max_scale=300.0, min_scale=0.1,
                 start_scale=None, expand=False, scroll_to_scale_factor=0.001):

        super().__init__()
        self.main_control = None
        self.expand = expand
        content.scroll = None
        content.expand = False
        if isinstance(content, ft.Image):
            content.fit = ft.ImageFit.COVER  # cover the whole area even if stretching the image
        self.inner_content = content
        self.scroll_to_scale_factor = scroll_to_scale_factor
        self.padding_color = padding_color
        self.content_with_padding = None
        self.width = width
        self.height = height
        self.innerstack = None
        self.start_scale = start_scale
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.on_scroll_callback = on_scroll
        self.on_click_callback = on_click
        self.on_pan_update_callback = on_pan_update
        self.content_height = content_height
        self.content_width = content_width
        self.scale = 1.0 if start_scale is None else start_scale
        self.previous_scale = self.scale
        self.offset_x = 0
        self.offset_y = 0
        self.zoom_x = None
        # the x coordinate of the point within the content where the mouse was when the zoom
        # was triggered
        self.zoom_y = None
        # the proper implementation should scale up and down around this point
        # not the center or corner of the content
        self.border_x = None
        self.border_y = None
        self.viewport_height = None
        self.viewport_width = None

    def build(self):
        """Builds the control.

        :return: the main control of the pan and zoom
        """
        content_column = ft.Column(
            alignment=ft.MainAxisAlignment.CENTER,
            expand=self.expand,
            controls=[
                ft.Row(
                    controls=[self.inner_content],
                    expand=self.expand,
                    alignment=ft.MainAxisAlignment.CENTER
                    )]
            )
        self.content_with_padding = ft.Container(
            height=self.content_height,
            width=self.content_width,
            bgcolor=self.padding_color,
            expand=self.expand,
            content=content_column
        )
        self.innerstack = ft.Stack(
            controls=[self.content_with_padding, ft.GestureDetector(
                on_pan_update=self.on_pan_update,
                on_scroll=self.on_scroll_update,
                on_tap_up=self.click_content
            )],
            left=0,
            top=0,
            width=self.width,
            height=self.height,
            expand=self.expand
        )
        self.main_control = SizeAwareControl(
            content=ft.Stack(controls=[self.innerstack]),
            expand=self.expand,
            on_resize=self.content_resize,
            width=self.width,
            height=self.height
        )
        self.controls = [self.main_control]

    def reset_content_dimensions(self):
        """Resets the content dimensions.

        This method is called when the viewport size changes.
        """
        self.scale = None
        self.update_content_pos_and_scale()

    def update_content_pos_and_scale(self):
        """Updates the position and scale of the content.

        This method is called when any parameter size, scale or position changes.
        It calculates the new position and sets it on the content.
        """
        if (self.viewport_width is None or self.viewport_height is None or
                self.viewport_height == 0 or self.content_height == 0):
            return
        viewport_ratio = self.viewport_width / self.viewport_height
        content_ratio = self.content_width / self.content_height
        # we want to pad the image with a border on the sides or on top and
        # below so that the resulting object has the same ratio as the viewport.
        # This is necessary to avoid inactive zones on the sides or top and bottom
        # (pan should work everywhere, not only on the image itself if it is very wide or tall)
        if viewport_ratio > content_ratio:
            # viewport is wider than image, so we pad the image with a border
            self.border_x = (self.content_height * viewport_ratio) - self.content_width
            self.border_y = 0
        else:
            # viewport is taller than image, so we pad the image with a border on the top and bottom
            # note: it is possible that both borders are zero
            self.border_y = (self.content_width / viewport_ratio) - self.content_height
            self.border_x = 0

        self.calculate_scale()

        stack_width = (self.content_width + self.border_x) * self.scale
        # the width of the full content scaled including the border
        stack_height = (self.content_height + self.border_y) * self.scale
        # the height of the full content scaled including the border
        stack_overflow_x = max(stack_width - self.viewport_width, 0)
        # the amount of pixels that are outside the viewport for the stack (the range of offset_x)
        stack_overflow_y = max(stack_height - self.viewport_height, 0)
        # the amount of pixels that are outside the viewport for the stack (the range of offset_y)
        content_overflow_x = max(self.content_width * self.scale - self.viewport_width, 0)
        # the amount of content pixels that are outside the viewport (the range of offset_x)
        content_overflow_y = max(self.content_height * self.scale - self.viewport_height, 0)
        # the amount of content pixels that are outside the viewport (the range of offset_y)

        self.adjust_offset_with_zoom_point(stack_height, stack_width)

        # Let's figure out the valid range for offset_x and offset_y
        # Both are negative since we are aiming with the top left corner outside the viewport
        # We have to aim the whole stack, not just the content
        # We know the movement range, which is content_overflow_x and content_overflow_y
        balance_x = min(stack_overflow_x / 2, self.border_x * self.scale / 2)
        balance_y = min(stack_overflow_y / 2, self.border_y * self.scale / 2)
        self.offset_x = self.clamp(self.offset_x, -content_overflow_x - balance_x, -balance_x)
        self.offset_y = self.clamp(self.offset_y, -content_overflow_y - balance_y, -balance_y)
        self.inner_content.width = self.content_width * self.scale
        self.inner_content.height = self.content_height * self.scale
        # TODO In theory, using scale would be enough and might even scale text
        #  better than using width and height. But scaling had strange artifacts,
        #  and strange offsets with strange overlays and erratic behaviour
        # self.inner_content.offset = ft.Offset(x=-0.0, y=-0.0)
        # self.inner_content.scale = self.scale

        self.innerstack.width = stack_width
        self.innerstack.height = stack_height
        self.content_with_padding.width = stack_width
        self.content_with_padding.height = stack_height
        self.innerstack.left = self.offset_x
        self.innerstack.top = self.offset_y
        self.innerstack.update()

    def adjust_offset_with_zoom_point(self, stack_height, stack_width):
        """Adjusts the offset according to the zoom point at zoom event.

        :param stack_height: the height of the stack
        :param stack_width: the width of the stack

        """

        if self.scale != self.previous_scale:
            if self.zoom_x is not None and self.innerstack.width is not None:
                # we have a zoom point, so we want to zoom in on that point
                # (where the mouse is when zooming)
                # we calculate the amount of size change and then adjust the offsets to match
                # the zoom point to the same position in the new image
                prevstack_width = (self.content_width + self.border_x) * self.previous_scale
                prevstack_height = (self.content_height + self.border_y) * self.previous_scale
                size_delta_x = stack_width - prevstack_width
                size_delta_y = stack_height - prevstack_height
                of_x = size_delta_x * (self.zoom_x / self.innerstack.width)  # offset of offset_x
                of_y = size_delta_y * (self.zoom_y / self.innerstack.height)  # offset of offset_y
                self.offset_x -= of_x  # offset is negative or zero since 0,0 is top left and we
                # want to move the content to the left and up only
                self.offset_y -= of_y
                self.zoom_x = None
                self.zoom_y = None
            self.previous_scale = self.scale

    def calculate_scale(self):
        """Calculates the scale of the content.

        The scale is calculated so that the content fits in the viewport with padding.
        """
        minimum_scale = min(
            self.viewport_width / (self.content_width + self.border_x),
            self.viewport_height / (self.content_height + self.border_y)
        )
        # we can't zoom out more than the full image in the viewport
        if self.scale is None:
            self.scale = self.start_scale if self.start_scale is not None else minimum_scale
            # start_scale is the preferred value, but if it is None, use the fully zoomed out
        self.scale = self.clamp(self.scale, max(minimum_scale, self.min_scale), self.max_scale)

    def content_resize(self, event: ft.canvas.CanvasResizeEvent):
        """
        :type event: ft.canvas.CanvasResizeEvent
        :param event: the event that triggered the resize
        """
        self.viewport_width = event.width
        self.viewport_height = event.height
        self.reset_content_dimensions()

    def on_pan_update(self, event: ft.DragUpdateEvent):
        """
        :type event: ft.DragUpdateEvent
        :param event: the event that triggered the pan
        """
        self.offset_x += event.delta_x
        self.offset_y += event.delta_y
        self.update_content_pos_and_scale()
        if self.on_pan_update_callback is not None:
            self.on_pan_update_callback(event)

    def on_scroll_update(self, event: ft.ScrollEvent):
        """
        :type event: ft.ScrollEvent
        :param event: scroll event
        """
        self.scale = self.scale * (1 + (event.scroll_delta_y * self.scroll_to_scale_factor))
        self.zoom_x = event.local_x
        self.zoom_y = event.local_y
        self.update_content_pos_and_scale()
        if self.on_scroll_callback is not None:
            self.on_scroll_callback(event)

    def clamp(self, value: float, min_value: float, max_value: float) -> float:
        """Clamps the value between min_value and max_value."""
        return min_value if value < min_value else max_value if value > max_value else value

    def click_content(self, event: ft.ControlEvent):
        """Handles click events on the content.

        :type event: ft.ControlEvent
        :param event: click event
        """
        # we don't need to handle offset_x and y since they are relative to the control
        x = event.local_x / self.scale - self.border_x / 2
        y = event.local_y / self.scale - self.border_y / 2
        if self.on_click_callback is not None:
            if 0 <= x < self.content_width and 0 <= y < self.content_height:
                event.local_x = x
                event.local_y = y
                self.on_click_callback(event)
     
     
class VideoContainer(ft.Container):
    """This will show a video you choose."""
    def __init__(
            self,
            video_path: str,
            fps: int = 0,
            play_after_loading=True,
            video_frame_fit_type: ft.ImageFit = None,
            video_progress_bar=True,
            video_play_button=True,
            exec_after_full_loaded=None,
            only_show_cover=False,
            content=None,
            ref=None,
            key=None,
            width=None,
            height=None,
            left=None,
            top=None,
            right=None,
            bottom=None,
            expand=None,
            col=None,
            opacity=None,
            rotate=None,
            scale=None,
            offset=None,
            aspect_ratio=None,
            animate_opacity=None,
            animate_size=None,
            animate_position=None,
            animate_rotation=None,
            animate_scale=None,
            animate_offset=None,
            on_animation_end=None,
            tooltip=None,
            visible=None,
            disabled=None,
            data=None,
            padding=None,
            margin=None,
            alignment=None,
            bgcolor=None,
            gradient=None,
            blend_mode=ft.BlendMode.SCREEN,
            border=None,
            border_radius=None,
            image_src=None,
            image_src_base64=None,
            image_repeat=None,
            image_fit=None,
            image_opacity=1.0,#OptionalNumber = None,
            shape=None,
            clip_behavior=None,
            ink=None,
            animate=None,
            blur=None,
            shadow=None,
            url=None,
            url_target=None,
            theme=None,
            theme_mode=None,
            on_click=None,
            on_long_press=None,
            on_hover=None
    ):
        super().__init__(content, ref, key, width, height, left, top, right, bottom, expand, col, opacity, rotate,
                         scale, offset, aspect_ratio, animate_opacity, animate_size, animate_position, animate_rotation,
                         animate_scale, animate_offset, on_animation_end, tooltip, visible, disabled, data, padding,
                         margin, alignment, bgcolor, gradient, blend_mode, border, border_radius, image_src,
                         image_src_base64, image_repeat, image_fit, image_opacity, shape, clip_behavior, ink, animate,
                         blur, shadow, url, url_target, theme, theme_mode, on_click, on_long_press, on_hover)
        self.__cur_play_frame = 0
        self.__video_pause_button = None
        self.__video_play_button = None
        self.__video_is_play = False
        self.vid_duration = None
        self.fps = fps
        self.__video_is_full_loaded = None
        self.video_frames = None
        self.exec_after_full_loaded = exec_after_full_loaded
        if not os.path.isfile(video_path):
            raise FileNotFoundError("Cannot find the video at the path you set.")
        self.all_frames_of_video = []
        self.frame_length = 0
        self.__video_played = False
        self.video_progress_bar = video_progress_bar
        self.video_play_button = video_play_button
        if video_frame_fit_type is None:
            self.video_frame_fit_type = ft.ImageFit.CONTAIN
        self.__ui()
        if only_show_cover:
            self.read_video_cover(video_path)
            return
        if play_after_loading:
            print("Please wait the video is loading..\nThis will take a time based on your video size...")
            self.read_the_video(video_path)
        else:
            threading.Thread(target=self.read_the_video, args=[video_path], daemon=True).start()
        self.audio_path = None
        self.__audio_path = None
        self.get_video_duration(video_path)
        self.__frame_per_sleep = 1.0 / self.fps

    def show_play(self):
        self.__video_is_play = False
        self.__video_play_button.visible = True
        self.__video_pause_button.visible = False
        self.__video_play_button.update()
        self.__video_pause_button.update()

    def show_pause(self):
        self.__video_is_play = True
        self.__video_play_button.visible = False
        self.__video_pause_button.visible = True
        self.__video_play_button.update()
        self.__video_pause_button.update()

    def __ui(self):
        # the video tools control
        self.video_tool_stack = ft.Stack(expand=False)
        self.content = self.video_tool_stack
        self.image_frames_viewer = ft.Image(expand=True, visible=False, fit=self.video_frame_fit_type)
        self.video_tool_stack.controls.append(ft.Row([self.image_frames_viewer], alignment=ft.MainAxisAlignment.CENTER))
        self.__video_progress_bar = ft.Container(height=2, bgcolor=ft.Colors.BLUE_200)
        self.video_tool_stack.controls.append(ft.Row([self.__video_progress_bar], alignment=ft.MainAxisAlignment.START))

        def play_video(e):
            print(e)
            if self.__video_is_play:
                self.pause()
                self.show_play()
            else:
                self.show_pause()
                self.play()

        self.__video_play_button = ft.IconButton(
            icon=ft.Icons.SMART_DISPLAY,
            icon_color=ft.Colors.WHITE54,
            icon_size=60,
            data=0,
            style=ft.ButtonStyle(
                elevation=4,
            ),
            on_click=play_video,
            visible=True
        )
        self.__video_pause_button = ft.IconButton(
            icon=ft.Icons.PAUSE_PRESENTATION,
            icon_color=ft.Colors.WHITE54,
            icon_size=60,
            data=0,
            style=ft.ButtonStyle(
                elevation=4,
            ),
            on_click=play_video,
            visible=False
        )
        self.video_tool_stack.controls.append(
            ft.Container(
                content=ft.Row(
                    controls=[
                        self.__video_play_button,
                        self.__video_pause_button
                    ]
                ),
                padding=ft.padding.only(25, 10, 10, 10),
                left=0,
                bottom=0,
            ),
        )
        if not self.video_progress_bar:
            self.__video_progress_bar.visible = False
        if not self.video_play_button:
            self.__video_play_button.visible = False

    def update_video_progress(self, frame_number):
        if not self.video_progress_bar:
            return
        percent_of_progress = frame_number / self.video_frames * 1
        if self.width:
            self.__video_progress_bar.width = percent_of_progress * 1 * self.width
        else:
            self.__video_progress_bar.width = percent_of_progress * 1 * self.page.width
        if self.__video_progress_bar.page is not None:
            try:
                self.__video_progress_bar.update()
            except Exception as e:
                pattern = r"control with ID '(.*)' not found"
                match = re.search(pattern, e.args[0])
                if not match:
                    print(e)
                return

    def update(self):
        self.image_frames_viewer.fit = self.video_frame_fit_type
        self.__video_progress_bar.visible = self.video_progress_bar
        return super().update()

    def play(self):
        """Play the video. (it's not blocking, because its on thread)."""
        if self.page is None:
            raise Exception("The control must be on page first.")
        self.__video_played = True
        threading.Thread(target=self.__play, daemon=True).start()

    def __play(self):
        self.image_frames_viewer.visible = True
        num = self.__cur_play_frame
        video_frames_len = len(self.all_frames_of_video)
        for index, i in enumerate(self.all_frames_of_video[self.__cur_play_frame:-1]):
            if not self.__video_played:
                self.__cur_play_frame = self.__cur_play_frame + index
                break
            if index + self.__cur_play_frame == video_frames_len - 2:
                self.__cur_play_frame = 0
            threading.Thread(target=self.update_video_progress, args=[num], daemon=True).start()
            self.image_frames_viewer.src_base64 = i
            try:
                self.image_frames_viewer.update()
            except Exception as e:
                pattern = r"control with ID '(.*)' not found"
                match = re.search(pattern, e.args[0])
                if not match:
                    print(e)
                return
            time.sleep(self.__frame_per_sleep)
            num += 1
        self.show_play()

    def pause(self):
        self.__video_played = False

    def read_video_cover(self, video_path):
        video = cv2.VideoCapture(video_path)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        slice_frame_num = frame_count / 2
        video.set(cv2.CAP_PROP_POS_FRAMES, slice_frame_num)
        success, frame = video.read()
        _, buffer = cv2.imencode('.jpg', frame)
        encoded_frame = base64.b64encode(buffer).decode('utf-8')
        if self.image_frames_viewer.src_base64 is None:
            self.image_frames_viewer.src_base64 = encoded_frame
            self.image_frames_viewer.visible = True
            if self.image_frames_viewer.page is not None:
                self.image_frames_viewer.update()
        video.release()

    def read_the_video(self, video_path):
        video = cv2.VideoCapture(video_path)
        success, frame = video.read()
        while success:
            _, buffer = cv2.imencode('.jpg', frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
            self.all_frames_of_video.append(encoded_frame)
            if self.image_frames_viewer.src_base64 is None:
                self.image_frames_viewer.src_base64 = encoded_frame
                self.image_frames_viewer.visible = True
                if self.image_frames_viewer.page is not None:
                    self.image_frames_viewer.update()
            success, frame = video.read()
        video.release()
        self.__video_is_full_loaded = True
        if self.exec_after_full_loaded:
            self.exec_after_full_loaded()
        self.frame_length = len(self.all_frames_of_video)
        return self.all_frames_of_video

    def get_video_duration(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video file")
            return
        if self.fps == 0:
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = fps
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.video_frames = total_frames
        duration = total_frames / fps
        self.vid_duration = duration
        cap.release()

''' Sample alt Object format
class Component(ft.Row):
    def __init__(self):
        super().__init__()
        self.build()
    def search(self, e):
        pass
    def build(self):
        self.expand = True
        #self.table
        self.parameter = Ref[TextField]()
        return Column(controls=[])
class Main:
    def __init__(self):
        self.page = None
    def __call__(self, page: Page):
        self.page = page
        page.title = "Alternative Boot experiment"
        self.add_stuff()
    def add_stuff(self):
        self.page.add(Text("Some text", size=20), color=Colors.ON_PRIMARY_CONTAINER, bgcolor=Colors.PRIMARY_CONTAINER, height=45)
        self.page.update()
main = Main()'''


import math
import logging
from itertools import cycle
from typing import List, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageColor, ImageFont

import flet as ft
import flet.canvas as cv

# --- Configuration ---
DEFAULT_FONT_PATH = "arial.ttf"
BEZIER_APPROXIMATION_STEPS = 20

# --- Logging Setup ---
logger = logging.getLogger("flet_canvas2img")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.WARNING) # Change to INFO or DEBUG for more verbosity

# --- Pillow Lanczos filter compatibility ---
try:
    LANCZOS_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS_FILTER = Image.LANCZOS

# --- Bezier Curve Helper Functions ---
def _approximate_quadratic_bezier(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], 
                                  steps: int = BEZIER_APPROXIMATION_STEPS) -> List[Tuple[float, float]]:
    """Returns a list of points approximating a quadratic Bezier curve."""
    if steps <= 0: return [p0, p2]
    return [
        (
            (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0],
            (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
        )
        for t in (i / steps for i in range(steps + 1))
    ]

def _approximate_cubic_bezier(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], 
                              steps: int = BEZIER_APPROXIMATION_STEPS) -> List[Tuple[float, float]]:
    """Returns a list of points approximating a cubic Bezier curve."""
    if steps <= 0: return [p0, p3]
    return [
        (
            (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0],
            (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
        )
        for t in (i / steps for i in range(steps + 1))
    ]

# --- Dashed Line Helper Function ---
def _draw_dashed_line(draw_context: ImageDraw.ImageDraw, 
                      points_list: List[Tuple[float, float]], 
                      dash_pattern_list: List[float], 
                      fill_color: Tuple[int, int, int, int], 
                      stroke_width: int):
    """Draws a dashed line or polyline. The dash pattern resets for each segment of the polyline."""
    if not points_list or len(points_list) < 2 or stroke_width <= 0:
        return # Nothing to draw or invisible
    if not dash_pattern_list or len(dash_pattern_list) < 2:
        # Draw as solid if no valid dash pattern
        draw_context.line(points_list, fill=fill_color, width=stroke_width)
        return
    for i in range(len(points_list) - 1):
        p_start, p_end = points_list[i], points_list[i+1]
        dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
        segment_length = math.hypot(dx, dy)
        if segment_length == 0: continue
        norm_dx, norm_dy = dx / segment_length, dy / segment_length
        current_pos_on_segment = 0.0
        current_pattern_iter = cycle(dash_pattern_list)
        while current_pos_on_segment < segment_length:
            dash_on_len = next(current_pattern_iter, 0.0)
            dash_off_len = next(current_pattern_iter, 0.0)
            if dash_on_len <= 0 and dash_off_len <= 0:
                logger.debug(f"Dash pattern {dash_pattern_list} resulted in zero advancement. Breaking dash for segment.")
                break
            if dash_on_len > 0:
                actual_draw_len = min(dash_on_len, segment_length - current_pos_on_segment)
                if actual_draw_len > 0:
                    start_x = p_start[0] + norm_dx * current_pos_on_segment
                    start_y = p_start[1] + norm_dy * current_pos_on_segment
                    end_x = p_start[0] + norm_dx * (current_pos_on_segment + actual_draw_len)
                    end_y = p_start[1] + norm_dy * (current_pos_on_segment + actual_draw_len)
                    draw_context.line([(start_x, start_y), (end_x, end_y)], fill=fill_color, width=stroke_width)
            current_pos_on_segment += dash_on_len + dash_off_len

# --- Flet Paint Attribute Extraction Helper ---
def get_flet_paint_attributes(shape_paint: Optional[ft.Paint] = None) -> \
    Tuple[Tuple[int,int,int,int], float, ft.PaintingStyle, ft.StrokeCap, ft.StrokeJoin, Optional[List[float]]]:
    """
    Extracts Flet paint attributes with sensible defaults.
    Returns: tuple (pil_color_rgba, stroke_width, paint_style, stroke_cap, stroke_join, dash_pattern)
    """
    color_str: str = "black"
    stroke_width: float = 1.0
    paint_style: ft.PaintingStyle = ft.PaintingStyle.FILL
    stroke_cap: ft.StrokeCap = ft.StrokeCap.BUTT
    stroke_join: ft.StrokeJoin = ft.StrokeJoin.MITER
    dash_pattern: Optional[List[float]] = None
    if shape_paint:
        if shape_paint.color is not None: color_str = shape_paint.color
        if shape_paint.stroke_width is not None: stroke_width = float(shape_paint.stroke_width)
        if shape_paint.style is not None: paint_style = shape_paint.style
        if shape_paint.stroke_cap is not None: stroke_cap = shape_paint.stroke_cap
        if shape_paint.stroke_join is not None: stroke_join = shape_paint.stroke_join
        flet_dash_attr = getattr(shape_paint, 'dash_pattern', None)
        if isinstance(flet_dash_attr, list) and len(flet_dash_attr) >= 2:
            try:
                processed_dash = [float(d) for d in flet_dash_attr]
                if all(d >= 0 for d in processed_dash):
                    dash_pattern = processed_dash
                else:
                    logger.warning(f"Dash pattern contains negative values: {flet_dash_attr}. Using solid line.")
            except ValueError:
                logger.warning(f"Dash pattern contains non-numeric values: {flet_dash_attr}. Using solid line.")
        elif flet_dash_attr is not None:
            logger.warning(f"Invalid dash_pattern format: {flet_dash_attr}. Expected list of numbers. Using solid line.")
    try:
        pil_color_rgba: Tuple[int,int,int,int] = ImageColor.getcolor(color_str, "RGBA")
    except ValueError:
        if str(color_str).lower() == "transparent":
            pil_color_rgba = (0, 0, 0, 0)
        else:
            logger.warning(f"Could not parse color '{color_str}'. Defaulting to opaque black.")
            pil_color_rgba = ImageColor.getcolor("black", "RGBA")
    return pil_color_rgba, stroke_width, paint_style, stroke_cap, stroke_join, dash_pattern

def canvas2img(
    shapes: List[Any],
    width: int = 770,
    height: int = 640,
    bgcolor: Tuple[int, int, int, int] = (255, 255, 255, 255),
    can_save: bool = True,
    save_path: str = "output.png",
    supersampling_factor: float = 1.0,
) -> Optional[Image.Image]:
    """
    Renders a list of Flet canvas shapes to a PIL (Pillow) Image object.
    Supports: Line, Circle, Ellipse, Arc, Rect (including border_radius), 
    Polygon, Polyline, Image, Path (with LineTo, QuadraticTo, CubicTo, Close), 
    and basic Text rendering.
    Features include anti-aliasing via supersampling and dashed line patterns.

    :param shapes: A list of Flet canvas shape objects (e.g., `cv.Line`, `cv.Circle`).
    :param width: The target width of the final output image in pixels. Must be > 0.
    :param height: The target height of the final output image in pixels. Must be > 0.
    :param bgcolor: Background color of the image as an (R, G, B, A) tuple.
                   Defaults to opaque white. For transparent, use (R,G,B,0).
    :param can_save: If True, the generated image will be saved to `save_path`.
    :param save_path: The file path where the image will be saved if `can_save` is True.
    :param supersampling_factor: The factor for supersampling (e.g., 2.0 for 2x resolution).
                                 Values <= 1.0 result in no supersampling. Higher values produce smoother edges but increase processing time and memory.
    :return: A `PIL.Image.Image` object, or `None` if a critical error occurs (e.g., invalid dimensions).
    """
    if not isinstance(shapes, list):
        logger.error("`shapes` argument must be a list. Cannot generate image.")
        return None
    if width <= 0 or height <= 0:
        logger.error(f"Target width ({width}) and height ({height}) must be positive. Cannot generate image.")
        return None

    scale = float(supersampling_factor) if supersampling_factor > 1.0 else 1.0
    render_width = max(1, int(round(width * scale)))
    render_height = max(1, int(round(height * scale)))
    try:
        img = Image.new("RGBA", (render_width, render_height), bgcolor)
    except Exception as e:
        logger.error(f"Failed to create new PIL Image ({render_width}x{render_height}): {e}")
        return None
    draw = ImageDraw.Draw(img, "RGBA")

    # --- Shape Processing Loop ---
    for shape_idx, shape_obj in enumerate(shapes):
        if shape_obj is None:
            continue
        paint_attributes = getattr(shape_obj, 'paint', None)
        pil_color, stroke_w_orig, style, cap, join, dash_pattern_orig = get_flet_paint_attributes(paint_attributes)
        stroke_w_scaled = stroke_w_orig * scale
        pil_draw_stroke_width = max(1, int(round(stroke_w_scaled))) if stroke_w_scaled > 0 else 0
        dash_pattern_scaled = [max(0.5, d_val * scale) if d_val > 0 else 0 for d_val in dash_pattern_orig] if dash_pattern_orig else None
        pil_join_type_arg = "curve" if join == ft.StrokeJoin.ROUND else None
        # Note: ft.StrokeJoin.BEVEL is not directly supported by PIL's line `joint` parameter. Miter is the default behavior if `joint` is None or not "curve".

        # --- Line Shape ---
        if isinstance(shape_obj, cv.Line):
            x1s, y1s = float(shape_obj.x1) * scale, float(shape_obj.y1) * scale
            x2s, y2s = float(shape_obj.x2) * scale, float(shape_obj.y2) * scale
            if pil_draw_stroke_width > 0:
                if dash_pattern_scaled:
                    _draw_dashed_line(draw, [(x1s, y1s), (x2s, y2s)], dash_pattern_scaled, pil_color, pil_draw_stroke_width)
                else:
                    draw.line([(x1s, y1s), (x2s, y2s)], fill=pil_color, width=pil_draw_stroke_width)
                if cap == ft.StrokeCap.ROUND and stroke_w_scaled > 0.1: # Check scaled width for meaningful cap
                    r_cap = stroke_w_scaled / 2.0
                    draw.ellipse((x1s - r_cap, y1s - r_cap, x1s + r_cap, y1s + r_cap), fill=pil_color, outline=None)
                    draw.ellipse((x2s - r_cap, y2s - r_cap, x2s + r_cap, y2s + r_cap), fill=pil_color, outline=None)

        # --- Circle & Ellipse Shapes (Common logic factored) ---
        elif isinstance(shape_obj, (cv.Circle, cv.Ellipse)):
            bbox_coords: List[float]
            if isinstance(shape_obj, cv.Circle):
                center_x, center_y = float(shape_obj.x) * scale, float(shape_obj.y) * scale
                radius = float(shape_obj.radius) * scale
                if radius < 0: radius = 0 # Ensure non-negative radius
                bbox_coords = [center_x - radius, center_y - radius, center_x + radius, center_y + radius]
            else: # cv.Ellipse
                x_el, y_el = float(shape_obj.x) * scale, float(shape_obj.y) * scale
                w_el, h_el = float(shape_obj.width) * scale, float(shape_obj.height) * scale
                if w_el < 0: w_el = 0; # Ensure non-negative dimensions
                if h_el < 0: h_el = 0;
                bbox_coords = [x_el, y_el, x_el + w_el, y_el + h_el]
            fill_arg, outline_arg, width_arg_el = None, None, 0
            if style == ft.PaintingStyle.FILL:
                fill_arg = pil_color
            elif style == ft.PaintingStyle.STROKE:
                if pil_draw_stroke_width > 0:
                    outline_arg = pil_color
                    width_arg_el = pil_draw_stroke_width
            elif style == ft.PaintingStyle.STROKE_AND_FILL:
                fill_arg = pil_color
                if pil_draw_stroke_width > 0:
                    outline_arg = pil_color 
                    width_arg_el = pil_draw_stroke_width
            draw.ellipse(bbox_coords, fill=fill_arg, outline=outline_arg, width=width_arg_el)

        # --- Arc Shape ---
        elif isinstance(shape_obj, cv.Arc):
            x_arc, y_arc = float(shape_obj.x) * scale, float(shape_obj.y) * scale
            w_arc, h_arc = float(shape_obj.width) * scale, float(shape_obj.height) * scale
            if w_arc <0: w_arc = 0
            if h_arc <0: h_arc = 0
            bbox_arc = [x_arc, y_arc, x_arc + w_arc, y_arc + h_arc]
            start_angle_deg = math.degrees(shape_obj.start_angle)
            sweep_angle_deg = math.degrees(shape_obj.sweep_angle)
            end_angle_deg = start_angle_deg + sweep_angle_deg
            if style == ft.PaintingStyle.FILL or style == ft.PaintingStyle.STROKE_AND_FILL:
                draw.pieslice(bbox_arc, start=start_angle_deg, end=end_angle_deg, fill=pil_color, outline=None)
            if style == ft.PaintingStyle.STROKE or style == ft.PaintingStyle.STROKE_AND_FILL:
                if pil_draw_stroke_width > 0:
                    draw.arc(bbox_arc, start=start_angle_deg, end=end_angle_deg, fill=pil_color, width=pil_draw_stroke_width)
        
        # --- Rectangle Shape ---
        elif isinstance(shape_obj, cv.Rect):
            x_r, y_r = float(shape_obj.x) * scale, float(shape_obj.y) * scale
            w_r, h_r = float(shape_obj.width) * scale, float(shape_obj.height) * scale
            if w_r < 0: w_r = 0
            if h_r < 0: h_r = 0
            bbox_r = [x_r, y_r, x_r + w_r, y_r + h_r]
            fill_arg_r, outline_arg_r, width_arg_r = None, None, 0
            if style == ft.PaintingStyle.FILL: fill_arg_r = pil_color
            elif style == ft.PaintingStyle.STROKE:
                if pil_draw_stroke_width > 0: outline_arg_r, width_arg_r = pil_color, pil_draw_stroke_width
            elif style == ft.PaintingStyle.STROKE_AND_FILL:
                fill_arg_r = pil_color
                if pil_draw_stroke_width > 0: outline_arg_r, width_arg_r = pil_color, pil_draw_stroke_width
            br_flet = getattr(shape_obj, "border_radius", None)
            radius_pil = 0.0
            if isinstance(br_flet, ft.BorderRadius):
                # PIL's rounded_rectangle supports only a single uniform radius.
                # We'll use top_left and warn if others are different.
                tl_scaled = float(br_flet.top_left) * scale
                if all(abs(getattr(br_flet, corner, 0.0) * scale - tl_scaled) < 1e-6 for corner in ["top_right", "bottom_right", "bottom_left"]):
                    radius_pil = tl_scaled
                else:
                    logger.warning("Per-corner border_radius with different values provided. PIL uses a single uniform radius. Using top-left.")
                    radius_pil = tl_scaled
            elif isinstance(br_flet, (int, float)) and br_flet > 0:
                radius_pil = float(br_flet) * scale
            if radius_pil > 0.1: # Draw rounded if radius is significant
                draw.rounded_rectangle(bbox_r, radius=radius_pil, fill=fill_arg_r, outline=outline_arg_r, width=width_arg_r)
            else: # Draw sharp rectangle
                draw.rectangle(bbox_r, fill=fill_arg_r, outline=outline_arg_r, width=width_arg_r)

        # --- Polygon Shape ---
        elif isinstance(shape_obj, cv.Polygon):
            points_poly = [(float(pt_x) * scale, float(pt_y) * scale) for pt_x, pt_y in shape_obj.points]
            if len(points_poly) < 2: continue
            if style == ft.PaintingStyle.FILL:
                if len(points_poly) >= 3: draw.polygon(points_poly, fill=pil_color, outline=None)
            elif style == ft.PaintingStyle.STROKE:
                if pil_draw_stroke_width > 0:
                    closed_pts = points_poly + [points_poly[0]] if len(points_poly) >=2 else points_poly
                    draw.line(closed_pts, fill=pil_color, width=pil_draw_stroke_width, joint=pil_join_type_arg)
            elif style == ft.PaintingStyle.STROKE_AND_FILL:
                if len(points_poly) >= 3: draw.polygon(points_poly, fill=pil_color, outline=None)
                if pil_draw_stroke_width > 0:
                    closed_pts = points_poly + [points_poly[0]] if len(points_poly) >=2 else points_poly
                    draw.line(closed_pts, fill=pil_color, width=pil_draw_stroke_width, joint=pil_join_type_arg)

        # --- Polyline Shape ---
        elif isinstance(shape_obj, cv.Polyline):
            points_pline = [(float(pt_x) * scale, float(pt_y) * scale) for pt_x, pt_y in shape_obj.points]
            if len(points_pline) < 2 or pil_draw_stroke_width <= 0: continue
            if dash_pattern_scaled:
                _draw_dashed_line(draw, points_pline, dash_pattern_scaled, pil_color, pil_draw_stroke_width)
            else:
                draw.line(points_pline, fill=pil_color, width=pil_draw_stroke_width, joint=pil_join_type_arg)
            if cap == ft.StrokeCap.ROUND and stroke_w_scaled > 0.1:
                r_cap_pl = stroke_w_scaled / 2.0
                draw.ellipse((points_pline[0][0]-r_cap_pl, points_pline[0][1]-r_cap_pl, points_pline[0][0]+r_cap_pl, points_pline[0][1]+r_cap_pl), fill=pil_color, outline=None)
                if len(points_pline) > 1 and points_pline[-1] != points_pline[0]: # Avoid double cap on closed-loop polyline
                    draw.ellipse((points_pline[-1][0]-r_cap_pl, points_pline[-1][1]-r_cap_pl, points_pline[-1][0]+r_cap_pl, points_pline[-1][1]+r_cap_pl), fill=pil_color, outline=None)

        # --- Image Shape ---
        elif isinstance(shape_obj, cv.Image):
            try:
                img_pil_src: Optional[Image.Image] = None
                if isinstance(shape_obj.src, Image.Image): img_pil_src = shape_obj.src.copy()
                elif isinstance(shape_obj.src, str) and shape_obj.src: img_pil_src = Image.open(shape_obj.src)
                else:
                    logger.warning(f"cv.Image 'src' attribute (type: {type(shape_obj.src)}) is not a PIL Image or valid file path. Skipping.")
                    continue
                x_img, y_img = float(shape_obj.x) * scale, float(shape_obj.y) * scale
                w_img, h_img = float(shape_obj.width) * scale, float(shape_obj.height) * scale
                if w_img <= 0 or h_img <= 0: continue # Skip zero-dimension images
                img_pil_src = img_pil_src.resize((int(round(w_img)), int(round(h_img))), LANCZOS_FILTER)
                if img_pil_src.mode != 'RGBA': img_pil_src = img_pil_src.convert('RGBA')
                img.paste(img_pil_src, (int(round(x_img)), int(round(y_img))), mask=img_pil_src)
            except FileNotFoundError:
                logger.warning(f"cv.Image: File not found at path '{shape_obj.src}'. Skipping.")
            except Exception as e:
                logger.warning(f"Could not process or paste cv.Image (src: {shape_obj.src}): {e}. Skipping.")

        # --- Path Shape ---
        elif isinstance(shape_obj, cv.Path):
            subpaths_data: List[dict] = []
            current_path_pts: List[Tuple[float,float]] = []
            current_path_start_pt: Optional[Tuple[float,float]] = None
            last_known_pen_pos: Optional[Tuple[float,float]] = None
            for path_elem in shape_obj.elements:
                if isinstance(path_elem, cv.Path.MoveTo):
                    if current_path_pts: subpaths_data.append({"points": list(current_path_pts), "closed": False})
                    pt_m = (float(path_elem.x) * scale, float(path_elem.y) * scale)
                    current_path_pts = [pt_m]
                    current_path_start_pt = pt_m
                    last_known_pen_pos = pt_m
                elif not last_known_pen_pos: # Other elements require a current pen position
                    logger.debug("Path element (LineTo, etc.) found without preceding MoveTo. Skipping element.")
                    continue 
                elif isinstance(path_elem, cv.Path.LineTo):
                    pt_l = (float(path_elem.x) * scale, float(path_elem.y) * scale)
                    current_path_pts.append(pt_l)
                    last_known_pen_pos = pt_l
                elif isinstance(path_elem, cv.Path.QuadraticTo):
                    ctrl_q = (float(path_elem.x1)*scale, float(path_elem.y1)*scale)
                    end_q = (float(path_elem.x2)*scale, float(path_elem.y2)*scale)
                    current_path_pts.extend(_approximate_quadratic_bezier(last_known_pen_pos, ctrl_q, end_q)[1:])
                    last_known_pen_pos = end_q
                elif isinstance(path_elem, cv.Path.CubicTo):
                    ctrl1=(float(path_elem.x1)*scale, float(path_elem.y1)*scale)
                    ctrl2=(float(path_elem.x2)*scale, float(path_elem.y2)*scale)
                    end_c=(float(path_elem.x3)*scale, float(path_elem.y3)*scale)
                    current_path_pts.extend(_approximate_cubic_bezier(last_known_pen_pos, ctrl1, ctrl2, end_c)[1:])
                    last_known_pen_pos = end_c
                # cv.Path.ArcTo is complex, requires SVG arc to Bezier/lines. Not implemented.
                elif isinstance(path_elem, cv.Path.Close):
                    if current_path_pts and current_path_start_pt:
                        # Explicitly close by adding start point if not already there
                        if current_path_pts[-1] != current_path_start_pt:
                            current_path_pts.append(current_path_start_pt)
                        subpaths_data.append({"points": list(current_path_pts), "closed": True})
                    # Reset for a new subpath (which must start with MoveTo)
                    current_path_pts, current_path_start_pt, last_known_pen_pos = [], None, None
            if current_path_pts: # Add any final unclosed subpath
                subpaths_data.append({"points": list(current_path_pts), "closed": False})

            # Draw the processed path
            if style in (ft.PaintingStyle.FILL, ft.PaintingStyle.STROKE_AND_FILL):
                for sub_info in subpaths_data:
                    if len(sub_info["points"]) >= 3: # Polygon fill needs at least 3 points
                        draw.polygon(sub_info["points"], fill=pil_color, outline=None)
            
            if style in (ft.PaintingStyle.STROKE, ft.PaintingStyle.STROKE_AND_FILL) and pil_draw_stroke_width > 0:
                for sub_info in subpaths_data:
                    pts_list, is_closed = sub_info["points"], sub_info["closed"]
                    if len(pts_list) >= 2: # Line stroke needs at least 2 points
                        if dash_pattern_scaled:
                            _draw_dashed_line(draw, pts_list, dash_pattern_scaled, pil_color, pil_draw_stroke_width)
                        else:
                            draw.line(pts_list, fill=pil_color, width=pil_draw_stroke_width, joint=pil_join_type_arg)
                        
                        # Apply caps only to visually open ends of non-explicitly-closed subpaths
                        if cap == ft.StrokeCap.ROUND and not is_closed and stroke_w_scaled > 0.1:
                            r_cap_path = stroke_w_scaled / 2.0
                            # Cap at the start of the subpath's line segments
                            draw.ellipse((pts_list[0][0]-r_cap_path, pts_list[0][1]-r_cap_path, 
                                          pts_list[0][0]+r_cap_path, pts_list[0][1]+r_cap_path), fill=pil_color, outline=None)
                            # Cap at the end, only if it's a different point (meaning path has length)
                            if len(pts_list) > 1 and pts_list[-1] != pts_list[0]:
                                draw.ellipse((pts_list[-1][0]-r_cap_path, pts_list[-1][1]-r_cap_path, 
                                              pts_list[-1][0]+r_cap_path, pts_list[-1][1]+r_cap_path), fill=pil_color, outline=None)
        
        # --- Text Shape (Basic Implementation) ---
        elif isinstance(shape_obj, cv.Text):
            try:
                font_size = int(max(1, (shape_obj.size if shape_obj.size is not None else 10) * scale))
                pil_font = ImageFont.load_default() # Default fallback
                try:
                    # Attempt to load specified/default font
                    font_family = getattr(shape_obj, 'font_family', DEFAULT_FONT_PATH) # Use font_family if available
                    if not font_family : font_family = DEFAULT_FONT_PATH # Ensure a path if font_family is empty string
                    # PIL/Pillow font selection by weight/style from family name is complex
                    pil_font = ImageFont.truetype(font_family, font_size)
                except IOError:
                    logger.debug(f"Font '{font_family}' not found or invalid. Using PIL default for Text.")
                except Exception as e_font: # Catch other font loading issues
                    logger.warning(f"Error loading font '{font_family}' for Text: {e_font}. Using PIL default.")
                text_x_s, text_y_s = float(shape_obj.x) * scale, float(shape_obj.y) * scale
                
                # Basic Flet TextAlign to PIL anchor mapping
                # Note: Flet START/END depend on LTR/RTL, not handled here. JUSTIFY not supported by draw.text.
                flet_align = getattr(shape_obj, "text_align", ft.TextAlign.LEFT)
                pil_anchor = "lt" # default: left-top
                if flet_align == ft.TextAlign.CENTER: pil_anchor = "mt" # middle-top
                elif flet_align == ft.TextAlign.RIGHT: pil_anchor = "rt" # right-top
                # More precise alignment needs font metrics (getbbox/getlength) for 'middle' of text block.
                draw.text((text_x_s, text_y_s), shape_obj.text, font=pil_font, fill=pil_color, anchor=pil_anchor)
            except Exception as e_render_text:
                logger.warning(f"Error rendering cv.Text ('{getattr(shape_obj, 'text', '')}'): {e_render_text}")

        # --- Unhandled Paint Features (e.g., Shaders/Gradients) ---
        elif paint_attributes and getattr(paint_attributes, "shader", None):
            logger.warning(f"Gradient paint (shader) on shape type {type(shape_obj).__name__} is not supported by this PIL-based renderer.")
        
        # --- Unknown or Unimplemented Shape Type ---
        elif not isinstance(shape_obj, (cv.Line, cv.Circle, cv.Ellipse, cv.Arc, cv.Rect, cv.Polygon, cv.Polyline, cv.Image, cv.Path, cv.Text)):
            shape_type_str = getattr(shape_obj, 'type', type(shape_obj).__name__)
            logger.warning(f"Flet canvas shape type '{shape_type_str}' is not implemented. Skipping shape at index {shape_idx}.")

    # --- Final Downscaling for Supersampling ---
    if scale > 1.0 and (render_width != width or render_height != height):
        final_target_width, final_target_height = int(width), int(height)
        if final_target_width > 0 and final_target_height > 0:
            try:
                img = img.resize((final_target_width, final_target_height), LANCZOS_FILTER)
            except Exception as e_resize:
                logger.error(f"Error during final image resize: {e_resize}")
        else:
            logger.warning("Target width/height for resize is zero or negative. Skipping final resize.")

    if can_save:
        try:
            img.save(save_path, "PNG")
        except Exception as e:
            logger.error(f"Failed to save image to '{save_path}': {e}")

    return img

class _DrawState: # Helper for MaskPainterDialog
    x: float; y: float
    def __init__(self): self.x = 0; self.y = 0

class MaskPainterDialog:
    def __init__(self, page: ft.Page, 
                 init_image_path: str, 
                 on_save_callback, # Function to call with mask_path on save
                 default_stroke_width: int = 40,
                 mask_suffix: str = "-mask",
                 supersampling: float = 4.0):
        self.page = page
        self.init_image_path = init_image_path
        self.on_save_callback = on_save_callback
        self.default_stroke_width = default_stroke_width
        self.current_stroke_width = default_stroke_width
        self.mask_suffix = mask_suffix
        self.supersampling = supersampling

        self._draw_state = _DrawState()
        self._image_pil = None
        self.image_width: int = 400
        self.image_height: int = 485

        # --- Dialog Controls ---
        self.bg_img = ft.Image(
            fit=ft.ImageFit.CONTAIN,
            width=self.image_width,
            height=self.image_height
        )
        self.canvas = cv.Canvas(
            content=ft.GestureDetector(
                on_pan_start=self._pan_start,
                on_pan_update=self._pan_update,
                drag_interval=10,
            ),
        )
        self.canvas_container = ft.Container(
            ft.Stack([self.bg_img, self.canvas]),
            border=ft.border.all(1, ft.Colors.OUTLINE),
            width=self.image_width,
            height=self.image_height,
            alignment=ft.alignment.center,
        )
        self.stroke_slider = ft.Slider(
            min=1, max=150, divisions=149, round=0, expand=True,
            label="{value}px", value=self.default_stroke_width,
            on_change=self._change_stroke,
            tooltip="Brush Stroke Size/Width",
        )
        self.clear_button = ft.IconButton(icon=ft.Icons.CLEAR, tooltip="Clear Mask", on_click=self._clear_canvas)
        self.dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text("Draw Mask"),
            content=ft.Column(
                [
                    self.canvas_container,
                    ft.Row([self.clear_button, self.stroke_slider]),
                ],
                tight=True, # Make column fit content
                scroll=ft.ScrollMode.ADAPTIVE # If content might overflow
            ),
            actions=[
                ft.TextButton("Cancel", on_click=self._close_dialog),
                ft.ElevatedButton("Save Mask ", on_click=self._save_mask_action),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            #on_dismiss=lambda e: print("Mask dialog dismissed"), # Optional
        )

    def _load_image_and_setup(self):
        try:
            if not self.init_image_path or not os.path.exists(self.init_image_path):
                self.page.show_snack_bar(ft.SnackBar(ft.Text(f"Error: Initial image not found at {self.init_image_path}"), open=True))
                self.bg_img.src = None # Or a placeholder image data_url
                self.image_width, self.image_height = 400, 300 # Default
            else:
                self._image_pil = PILImage.open(self.init_image_path)
                self.image_width, self.image_height = self._image_pil.size
                self.bg_img.src = self.init_image_path
            
            self.bg_img.width = self.image_width
            self.bg_img.height = self.image_height
            self.canvas_container.width = self.image_width
            self.canvas_container.height = self.image_height
            self.canvas.shapes.clear()
            #self.bg_img.update()
            #self.canvas_container.update()
            #self.canvas.update()

        except Exception as e:
            print(f"Error loading image for mask dialog: {e}")
            self.bg_img.src = None 
            self.image_width, self.image_height = 400, 300
            #self.bg_img.update()
            #self.canvas_container.update()
    
    def _pan_start(self, e: ft.DragStartEvent):
        self._draw_state.x = e.local_x
        self._draw_state.y = e.local_y

    def _pan_update(self, e: ft.DragUpdateEvent):
        paint = ft.Paint(
            stroke_width=self.current_stroke_width,
            color=ft.Colors.BLACK,
            stroke_join=ft.StrokeJoin.ROUND,
            stroke_cap=ft.StrokeCap.ROUND,
            style=ft.PaintingStyle.STROKE # Important for lines
        )
        self.canvas.shapes.append(cv.Line(self._draw_state.x, self._draw_state.y, e.local_x, e.local_y, paint=paint))
        self.canvas.update()
        self._draw_state.x = e.local_x
        self._draw_state.y = e.local_y

    def _clear_canvas(self, e=None): # e can be None if called internally
        self.canvas.shapes.clear()
        self.canvas.update()

    def _change_stroke(self, e: ft.ControlEvent):
        self.current_stroke_width = int(e.control.value)

    def _generate_mask_path(self) -> str:
        base, ext = os.path.splitext(os.path.basename(self.init_image_path))
        mask_filename = f"{base}{self.mask_suffix}.png"        
        save_dir = os.path.dirname(self.init_image_path)
        if not save_dir: # If init_image_path was just a filename
            save_dir = getattr(self.page, 'uploads_dir', 'uploads') 
            os.makedirs(save_dir, exist_ok=True)
        full_mask_path = os.path.join(save_dir, mask_filename)
        count = 1
        while os.path.exists(full_mask_path):
            mask_filename = f"{base}{self.mask_suffix}-{count}.png"
            full_mask_path = os.path.join(save_dir, mask_filename)
            count += 1
        return full_mask_path

    def _save_mask_action(self, e: ft.ControlEvent):
        if not self.canvas.shapes:
            self.page.show_snack_bar(ft.SnackBar(ft.Text("Nothing to save, mask is empty."), open=True))
            return
        mask_file_path = self._generate_mask_path()
        try:
            #from aeionic_components import canvas2img # Or wherever it is
            canvas2img(
                self.canvas.shapes,
                width=self.image_width,
                height=self.image_height,
                bgcolor=(255, 255, 255, 255), #(0, 0, 0, 0),  # Transparent background for the mask strokes
                can_save=True,
                save_path=mask_file_path,
                supersampling_factor=self.supersampling
            )
            if self.on_save_callback:
                self.on_save_callback(mask_file_path)
            self._close_dialog()
        except ImportError:
            print("ERROR: canvas2img function not found. Please ensure it's correctly imported.")
        except Exception as ex:
            self.page.show_snack_bar(ft.SnackBar(ft.Text(f"Error saving mask: {ex}"), open=True))

    def _close_dialog(self, e=None): # e can be None if called internally
        self.dialog.open = False
        self.page.update()

    def open(self):
        self._load_image_and_setup() # Load image and set dimensions
        self.page.overlay.append(self.dialog)
        self.dialog.open = True
        self.page.update()



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

from PIL import Image, ImageDraw, ImageColor
import math

def get_flet_paint_attributes(shape_paint: ft.Paint = None):
    """
    Helper function to extract and provide default Flet paint attributes.
    Returns RGBA color tuple for PIL.
    """
    # Default Flet Paint properties
    color_str = "black"
    stroke_width = 1.0
    paint_style = ft.PaintingStyle.FILL
    stroke_cap = ft.StrokeCap.BUTT
    stroke_join = ft.StrokeJoin.MITER

    if shape_paint:
        if shape_paint.color is not None:
            color_str = shape_paint.color
        if shape_paint.stroke_width is not None:
            stroke_width = shape_paint.stroke_width
        if shape_paint.style is not None:
            paint_style = shape_paint.style
        if shape_paint.stroke_cap is not None:
            stroke_cap = shape_paint.stroke_cap
        if shape_paint.stroke_join is not None:
            stroke_join = shape_paint.stroke_join

    try:
        pil_color_rgba = ImageColor.getcolor(color_str, "RGBA")
    except ValueError:
        if color_str.lower() == "transparent":
            pil_color_rgba = (0, 0, 0, 0)
        else:
            print(f"Warning: Could not parse color '{color_str}'. Defaulting to black.")
            pil_color_rgba = ImageColor.getcolor("black", "RGBA")

    return pil_color_rgba, float(stroke_width), paint_style, stroke_cap, stroke_join


def canvas2img(shapes: list, width: int = 770, height: int = 640,
               bgcolor: tuple = (255, 255, 255, 255),
               can_save: bool = True, save_path: str = "output.png",
               supersampling_factor: float = 1.0):
    """
    Converts Flet canvas shapes to a PIL Image with optional supersampling for anti-aliasing.

    :param shapes: List of shape objects from a Flet canvas.
    :param width: Width of the final output image.
    :param height: Height of the final output image.
    :param bgcolor: Background color of the image (R, G, B, A).
    :param can_save: If True, saves the image.
    :param save_path: Path to save the generated image.
    :param supersampling_factor: Factor for supersampling (e.g., 2 for 2x SSAA).
                                 1 means no supersampling.
    :return: PIL Image object.
    """

    if supersampling_factor <= 1.0:
        scale = 1.0
        render_width = int(width)
        render_height = int(height)
    else:
        scale = float(supersampling_factor)
        render_width = int(width * scale)
        render_height = int(height * scale)
        if render_width == 0 or render_height == 0: # Safety for tiny target dimensions
            scale = 1.0
            render_width = int(width)
            render_height = int(height)


    img = Image.new("RGBA", (render_width, render_height), bgcolor)
    draw = ImageDraw.Draw(img, "RGBA")

    for shape in shapes:
        pil_color, stroke_w_orig, style, cap, join = get_flet_paint_attributes(getattr(shape, 'paint', None))

        # Scale stroke width for supersampling
        scaled_stroke_w = stroke_w_orig * scale
        if stroke_w_orig > 0:
            # Ensure visible lines are at least 1px in the render target
            # but allow 0 if original was 0 (e.g. for filled shapes with no outline intent)
            pil_draw_stroke_width = max(1, int(round(scaled_stroke_w))) if scaled_stroke_w > 0 else 0
        else:
            pil_draw_stroke_width = 0


        if isinstance(shape, cv.Line):
            # Scale coordinates
            scaled_x1, scaled_y1 = float(shape.x1) * scale, float(shape.y1) * scale
            scaled_x2, scaled_y2 = float(shape.x2) * scale, float(shape.y2) * scale

            draw.line([(scaled_x1, scaled_y1), (scaled_x2, scaled_y2)],
                      fill=pil_color, width=pil_draw_stroke_width)

            if cap == ft.StrokeCap.ROUND and scaled_stroke_w > 0:
                r_cap = scaled_stroke_w / 2.0
                if r_cap > 0: # Only draw caps if they have some radius
                    draw.ellipse((scaled_x1 - r_cap, scaled_y1 - r_cap, scaled_x1 + r_cap, scaled_y1 + r_cap),
                                 fill=pil_color, outline=None)
                    draw.ellipse((scaled_x2 - r_cap, scaled_y2 - r_cap, scaled_x2 + r_cap, scaled_y2 + r_cap),
                                 fill=pil_color, outline=None)

        elif isinstance(shape, cv.Circle):
            scaled_x = float(shape.x) * scale
            scaled_y = float(shape.y) * scale
            scaled_radius = float(shape.radius) * scale
            
            bbox = [scaled_x - scaled_radius, scaled_y - scaled_radius,
                    scaled_x + scaled_radius, scaled_y + scaled_radius]
            
            fill_arg, outline_arg, width_arg = None, None, 0
            if style == ft.PaintingStyle.FILL:
                fill_arg = pil_color
            elif style == ft.PaintingStyle.STROKE:
                outline_arg = pil_color
                width_arg = pil_draw_stroke_width
            elif style == ft.PaintingStyle.STROKE_AND_FILL:
                fill_arg = pil_color
                outline_arg = pil_color 
                width_arg = pil_draw_stroke_width
            
            draw.ellipse(bbox, fill=fill_arg, outline=outline_arg, width=width_arg)

        elif isinstance(shape, cv.Arc):
            scaled_x, scaled_y = float(shape.x) * scale, float(shape.y) * scale
            scaled_arc_w, scaled_arc_h = float(shape.width) * scale, float(shape.height) * scale
            
            bbox = [scaled_x, scaled_y, scaled_x + scaled_arc_w, scaled_y + scaled_arc_h]
            start_angle_deg = math.degrees(shape.start_angle)
            sweep_angle_deg = math.degrees(shape.sweep_angle)
            end_angle_deg = start_angle_deg + sweep_angle_deg
            
            draw.arc(bbox, start=start_angle_deg, end=end_angle_deg,
                     fill=pil_color, width=pil_draw_stroke_width)
            # Round caps for arcs are complex to add manually here.

        elif isinstance(shape, cv.Rect):
            scaled_x, scaled_y = float(shape.x) * scale, float(shape.y) * scale
            scaled_rect_w, scaled_rect_h = float(shape.width) * scale, float(shape.height) * scale
            bbox = [scaled_x, scaled_y, scaled_x + scaled_rect_w, scaled_y + scaled_rect_h]

            fill_arg, outline_arg, width_arg = None, None, 0
            if style == ft.PaintingStyle.FILL:
                fill_arg = pil_color
            elif style == ft.PaintingStyle.STROKE:
                outline_arg = pil_color
                width_arg = pil_draw_stroke_width
            elif style == ft.PaintingStyle.STROKE_AND_FILL:
                fill_arg = pil_color
                outline_arg = pil_color
                width_arg = pil_draw_stroke_width
            
            scaled_br_val = 0
            if hasattr(shape, 'border_radius') and shape.border_radius:
                br_val_orig = 0
                if isinstance(shape.border_radius, (int, float)):
                    br_val_orig = float(shape.border_radius)
                elif isinstance(shape.border_radius, ft.BorderRadius):
                    br_val_orig = float(shape.border_radius.top_left) # Simplification
                scaled_br_val = br_val_orig * scale

            if scaled_br_val > 0:
                draw.rounded_rectangle(bbox, radius=scaled_br_val, fill=fill_arg,
                                       outline=outline_arg, width=width_arg)
            else:
                draw.rectangle(bbox, fill=fill_arg, outline=outline_arg, width=width_arg)
        
        elif isinstance(shape, cv.Path):
            pil_join_type = "curve" if join == ft.StrokeJoin.ROUND else None
            
            current_subpath_points = []
            path_start_point_for_close = None

            for element in shape.elements:
                if isinstance(element, cv.Path.MoveTo):
                    if len(current_subpath_points) >= 2:
                        draw.line(current_subpath_points, fill=pil_color, 
                                  width=pil_draw_stroke_width, joint=pil_join_type)
                        if cap == ft.StrokeCap.ROUND and scaled_stroke_w > 0: # cap for open path
                            r_cap = scaled_stroke_w / 2.0
                            if r_cap > 0:
                                ps_cap, pe_cap = current_subpath_points[0], current_subpath_points[-1]
                                draw.ellipse((ps_cap[0]-r_cap, ps_cap[1]-r_cap, ps_cap[0]+r_cap, ps_cap[1]+r_cap), fill=pil_color)
                                draw.ellipse((pe_cap[0]-r_cap, pe_cap[1]-r_cap, pe_cap[0]+r_cap, pe_cap[1]+r_cap), fill=pil_color)
                    
                    scaled_ex, scaled_ey = float(element.x) * scale, float(element.y) * scale
                    current_subpath_points = [(scaled_ex, scaled_ey)]
                    path_start_point_for_close = current_subpath_points[0]
                
                elif isinstance(element, cv.Path.LineTo):
                    scaled_ex, scaled_ey = float(element.x) * scale, float(element.y) * scale
                    current_subpath_points.append((scaled_ex, scaled_ey))
                
                elif isinstance(element, cv.Path.Close):
                    if path_start_point_for_close and current_subpath_points:
                        if current_subpath_points[-1] != path_start_point_for_close:
                             current_subpath_points.append(path_start_point_for_close)
                    if len(current_subpath_points) >= 2:
                        draw.line(current_subpath_points, fill=pil_color, 
                                  width=pil_draw_stroke_width, joint=pil_join_type)
                    current_subpath_points = []
                    path_start_point_for_close = None

                elif isinstance(element, cv.Path.QuadraticTo):
                    # Scaled control point, scaled end point
                    # scaled_c1x, scaled_c1y = float(element.x1) * scale, float(element.y1) * scale
                    scaled_ex, scaled_ey = float(element.x2) * scale, float(element.y2) * scale
                    current_subpath_points.append((scaled_ex, scaled_ey)) # Approximation
                elif isinstance(element, cv.Path.CubicTo):
                    # Scaled control points, scaled end point
                    # scaled_c1x, scaled_c1y = float(element.x1) * scale, float(element.y1) * scale
                    # scaled_c2x, scaled_c2y = float(element.x2) * scale, float(element.y2) * scale
                    scaled_ex, scaled_ey = float(element.x3) * scale, float(element.y3) * scale
                    current_subpath_points.append((scaled_ex, scaled_ey)) # Approximation
            
            if len(current_subpath_points) >= 2:
                draw.line(current_subpath_points, fill=pil_color, 
                          width=pil_draw_stroke_width, joint=pil_join_type)
                if cap == ft.StrokeCap.ROUND and scaled_stroke_w > 0: # cap for final open path
                    r_cap = scaled_stroke_w / 2.0
                    if r_cap > 0:
                        ps_cap, pe_cap = current_subpath_points[0], current_subpath_points[-1]
                        draw.ellipse((ps_cap[0]-r_cap, ps_cap[1]-r_cap, ps_cap[0]+r_cap, ps_cap[1]+r_cap), fill=pil_color)
                        draw.ellipse((pe_cap[0]-r_cap, pe_cap[1]-r_cap, pe_cap[0]+r_cap, pe_cap[1]+r_cap), fill=pil_color)

    # Downscale if supersampling was used
    if scale > 1.0 and width > 0 and height > 0:
        try:
            # For Pillow >= 8.0.0
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            # For older Pillow versions
            resample_filter = Image.LANCZOS
        img = img.resize((int(width), int(height)), resample_filter)

    if can_save:
        img.save(save_path, "PNG")
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
                 supersampling: float = 2.0):
        
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
            on_change=self._change_stroke
        )
        self.clear_button = ft.IconButton(
            icon=ft.Icons.CLEAR, tooltip="Clear Mask", on_click=self._clear_canvas
        )
        
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
                ft.ElevatedButton("Save Mask", on_click=self._save_mask_action),
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
        self.canvas.shapes.append(
            cv.Line(self._draw_state.x, self._draw_state.y, e.local_x, e.local_y, paint=paint)
        )
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

        # Simple available_file logic (add numbering if exists)
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


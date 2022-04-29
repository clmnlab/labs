
import psychopy
from threading import Timer
import queue
from Module.sj_file_system import CsvManager

import time
from psychopy import core



"""
Note: 
★ psychopy position

    y axis
        1
        0.5
        0(center)
        -0.5
        1

    x axis: -1 -0.5 0(center) 0.5 1

★ psychopy align
psychopy's absolute positioning is that (0.0) is center
psychopy's object positioning is depending on object's anchor

when a object's position is (0,0) then psychopy match the object's anchor and psychopy's absolution position
if the anchor is left(left boundar), the anchor point is matched to center position
so, the left side point of box's position is 0
"""

class Direct_fire_timer:
    """
    This class is made for call certain procedure when the time is reached by timer or when need to call directly
    """
    def start(self, seconds, proc):
        self.proc = proc
        self.timer = Timer(seconds, proc)
        self.timer.start()
    def cancel(self):
        self.timer.cancel()
        self.timer = None
    def direct_proc(self):
        self.cancel()
        self.proc()

class St_Unit:
    # It is abstract type of Stimulus
    def __init__(self, showing_time):
        self.showing_time = showing_time

class Image_st_unit(St_Unit):
    # It shows image stimulus
    def __init__(self, image_path, unit = None, size = None, showing_time = 0.0):
        """
        :param image_path: image path for loading
        :param size: image size (width, height) ex) (30, 30)
        :param unit: size unit ex) None, ‘norm’, ‘cm’, ‘deg’, ‘degFlat’, ‘degFlatPos’, or ‘pix’.
        :param showing_time: stimulus showing time
        """
        super().__init__(showing_time)
        self.image_path = image_path
        self.size = size
        self.unit = unit

class Text_st_unit(St_Unit):
    # It shows text stimulus
    def __init__(self,
                 text,
                 color=[0,0,0],
                 showing_time = 0.0,
                 text_height=0.3,
                 font=None,
                 fontFiles=None):
        super().__init__(showing_time)
        self.text = text
        self.color = color
        self.text_height = text_height
        self.font = font
        self.fontFiles = fontFiles

class Sequence_st_text_unit(St_Unit):
    #: ex) 3-2-1-4-3
    def __init__(self,
                 text_units,
                 showing_time = 0.0,
                 color = [1,1,1],
                 text_height=0.3,
                 is_count_correct=False,
                 unit_correct_count = 0,
                 correct_count = 0,
                 is_display_unit_correct = False):
        self.text_units = text_units
        self.showing_time = showing_time
        self.color = color
        self.text_height = text_height
        self.is_count_correct = is_count_correct
        if self.is_count_correct:
            self.is_display_unit_correct = is_display_unit_correct
            self.unit_correct_count = unit_correct_count # increase the count when single text_unit is matched correctly
            self.correct_count = correct_count # increase the count when full sequence is matched correctly

        type_check = [isinstance(t_u, Text_st_unit) for t_u in text_units]
        if sum(type_check) == len(text_units):
            result = []
            for unit in text_units:
                result += unit.text
            self.texts = result
        else:
            type_check = [isinstance(t, str) for t in text_units]
            if sum(type_check) == len(text_units):
                self.texts = text_units
            else:
                raise Exception("Type Error!")

    def __str__(self):
        return str(self.texts)

class ISI_st_unit(St_Unit):
    # It shows ISI stimulus
    def __init__(self, text, color=[0,0,0], showing_time = 0.0, text_height=0.3):
        super().__init__(showing_time)
        self.text = text
        self.color = color
        self.text_height = text_height

class BundleInterval_st_unit(St_Unit):
    # It shows bundle interval stimulus
    def __init__(self, text, color=[0,0,0], showing_time = 0.0, text_height=0.3):
        super().__init__(showing_time)
        self.text = text
        self.color = color
        self.text_height = text_height

class St_bundle:
    # What is bundle?: bundle means that a box consists of many pairs of stimulus and rest state
    # It consists of many unit stimuluses
    def __init__(self, units):
        self.units = units
        self.ISI_units = []

class Text_st_bundle(St_bundle):
    # It consists of many Text stimulus
    def __init__(self, units, ISI_times):
        """
        :param units: list of Text_st_unit
        :param ISI_avg_time: average time(Secs) for ISI. if this value is 3 the ISI average time is 3sec per bundle
        """
        # units: list of Text_st_unit
        self.units = units
        self.ISI_times = ISI_times + [0]

        ISIs_stimulus = []
        for i in range(0, len(units)):
            ISIs_stimulus.append(ISI_st_unit("+", showing_time=self.ISI_times[i]))
        self.ISI_units = ISIs_stimulus

class Sequence_st_bundle(St_bundle):
    # It consists of many Sequence stimulus
    def __init__(self, sequences, ISI_interval):
        super().__init__(sequences)
        for i in range(0, len(sequences)):
            self.ISI_units.append(ISI_st_unit("+", showing_time=ISI_interval))

    def __str__(self):
        print_value = ""
        for unit in self.units:
            print_value += unit.__str__() + " "

        return print_value

class St_Package:
    # set of bundles
    # It consists of many bundles
    def __init__(self, bundles, bundle_intervals, interval_text, start_wait_seconds = 0):
        self.start_wait_seconds = start_wait_seconds
        self.bundles = bundles
        self.bundle_intervals = bundle_intervals

        self.start_unit = ISI_st_unit("+", showing_time=start_wait_seconds)

        interval_units = []
        for i in range(0, len(bundles)):
            remain = i % len(bundle_intervals)
            if bundle_intervals[remain] != 0:
                interval_units.append(BundleInterval_st_unit(interval_text, showing_time=bundle_intervals[remain]))
        self.interval_units = interval_units

    def __str__(self):
        print_value = ""
        for bundle in self.bundles:
            print(bundle.__str__())
            print_value += bundle.__str__() + " "

        return print_value

class Intermediater:
    def __init__(self):
        print("Init Intermeidater")

        # essential property (must need to be set)
        self.input_interface_manager = None
        self.display_manager = None
        self.event_manager = None

    def get_win_size(self):
        return self.display_manager.get_win_size()

    def set_input_interface_manager(self, input_interface_manager):
        self.input_interface_manager = input_interface_manager

    def set_display_manager(self, display_manager):
        self.display_manager = display_manager

    def set_event_manager(self, event_manager):
        self.event_manager = event_manager

    def one_input_only(self):
        if self.event_manager != None:
            self.event_manager.set_is_activate_one_input(True)
            self.event_manager.set_is_activate_multiple_input(False, 0)
        else:
            print("Event manager is not loaded")

    def one_multi_input_both(self, multi_input_count):
        if self.event_manager != None:
            self.event_manager.set_is_activate_one_input(True)
            self.event_manager.set_is_activate_multiple_input(is_multiple_activate=True,
                                                              target_input_count=multi_input_count)
        else:
            print("Event manager is not loaded")

    def invalid_event(self):
        if self.event_manager != None:
            self.event_manager.set_is_activate_one_input(False)
            self.event_manager.set_is_activate_multiple_input(False, 0)
        else:
            print("Event manager is not loaded")

    def listen_key_input(self, input):
        if self.event_manager != None:
            self.event_manager.listen_key_input(input, self.display_manager.current_showing_stimlus)
        else:
            print("Event manager is not loaded")

    def listen_mouse_input(self, mousePos):
        if self.event_manager != None:
            self.event_manager.listen_mouse_input(mousePos, self.display_manager.current_showing_stimlus)
        else:
            print("Event manager is not loaded")

    def set_valid_keys(self, keys):
        self.valid_keys = keys

    def wait_start(self):
        self.input_interface_manager.wait_start()

    def insert_interrupt(self, interrupt):
        self.input_interface_manager.set_interrupt_operation(interrupt)
        self.input_interface_manager.insert_interrupt(True)

    def is_use_event_manager(self):
        return False if self.event_manager == None else True

class Psy_display_manager:
    def __init__(self, intermediater):
        self.intermediater = intermediater
        self.timer = Direct_fire_timer()
        self.current_step = -1 # It denotes that current step of stimulus
        self.current_showing_stimlus = None # currently showed stimulus in display
        self.total_stimuluses = None # It denotes that total stimulus needs to be showed
        self.stimulus_showing_handler = None # It is called when stimulus is showed
        self.is_loaded = False

    def open_window(self, size, color = [-1,-1,-1], is_full_screen = False):
        print("open window")
        from psychopy import visual, monitors
        monitors.Monitor("")
        self.visual = visual
        self.win = self.visual.Window(monitor=monitors.Monitor('exp monitor'),
                                      size=size,
                                      color=color,
                                      colorSpace='rgb',
                                      fullscr=is_full_screen,
                                      allowGUI=False)

        self.win_size = self.win.size
        self.win.mouseVisible = False

    def get_win_size(self):
        return self.win_size

    def set_stimulus_showing_handler(self, handler):
        self.stimulus_showing_handler = handler

    def wait_start(self, ready_keys, stop_keys, iteration = None, load_callback = None):
        """
        Wait start

        :param ready_keys: ready key. when this key is inputted, This class is ready for receiving start signal
        :param stop_keys: stop key for quiting psychopy
        :param iteration: run iteration
        :param load_callback: callback function when the psychopy loads completly
        """
        if self.is_loaded == False:
            self.is_loaded = True
            if load_callback != None:
                load_callback()

        # 원래는 interface manager에서 wait key를 하는게 맞으나... 편의상 삽입
        self.show_stimulus(Text_st_unit("+"))
        keys = psychopy.event.waitKeys(keyList=ready_keys + stop_keys)

        if keys[0] in ready_keys:
            self.show_stimulus(Text_st_unit("")) # str(iteration) + " trial" + " Ready"
            self.intermediater.wait_start()
            return True
        else:
            core.quit()
            self.win.close()

            return False

    def call_stimulus_showing_handler(self, type, stimulus, showing_time):
        if self.stimulus_showing_handler != None:
            self.stimulus_showing_handler(type, stimulus, showing_time)

    def show_stimulus(self, stimulus):
        self.current_showing_stimlus = stimulus

        if isinstance(stimulus, Image_st_unit):
            self.intermediater.one_input_only()

            print(str.format("showing stimulus: {0}, showing time: {1}sec", stimulus.image_path, stimulus.showing_time))
            image = self.visual.ImageStim(win=self.win,
                                          image=stimulus.image_path,
                                          units=stimulus.unit,
                                          size=stimulus.size)
            image.draw()
            self.win.flip()
            self.call_stimulus_showing_handler("image", stimulus.image_path, stimulus.showing_time)
        elif isinstance(stimulus, ISI_st_unit):
            text = self.visual.TextStim(win=self.win,
                                        text=stimulus.text,
                                        height=stimulus.text_height,
                                        bold=True,
                                        colorSpace="rgb",
                                        color=stimulus.color,)
            text.draw()
            self.win.flip()
            print(str.format("showing stimulus: {0}, showing time: {1}sec", stimulus.text, stimulus.showing_time))
            self.call_stimulus_showing_handler("ISI", stimulus.text, stimulus.showing_time)
        elif isinstance(stimulus, BundleInterval_st_unit):
            self.intermediater.one_input_only()
            text = self.visual.TextStim(win=self.win,
                                        text=stimulus.text,
                                        height=stimulus.text_height,
                                        bold=True,
                                        colorSpace="rgb",
                                        color=stimulus.color)
            text.draw()
            self.win.flip()
            print(str.format("showing stimulus: {0}, showing time: {1}sec", stimulus.text, stimulus.showing_time))
            self.call_stimulus_showing_handler("Bundle Interval", stimulus.text, stimulus.showing_time)
        elif isinstance(stimulus, Text_st_unit):
            self.intermediater.one_input_only()
            if stimulus.font != None:
                """
                print(stimulus.fontFiles)
                print(stimulus.font)
                """
                text = self.visual.TextStim(win=self.win,
                                            text=stimulus.text,
                                            height=stimulus.text_height,
                                            colorSpace="rgb",
                                            color=stimulus.color,
                                            font=stimulus.font,
                                            fontFiles=stimulus.fontFiles)
            else:
                text = self.visual.TextStim(win=self.win,
                                            text=stimulus.text,
                                            height=stimulus.text_height,
                                            bold=True,
                                            colorSpace="rgb",
                                            color=stimulus.color)

            text.draw()
            self.win.flip()
            print(str.format("showing stimulus: {0}, showing time: {1}sec", stimulus.text, stimulus.showing_time))
            self.call_stimulus_showing_handler("single text", stimulus.text, stimulus.showing_time)
        elif isinstance(stimulus, Sequence_st_text_unit):
            # self.intermediater.one_multi_input_both(len(stimulus.texts))
            self.intermediater.one_input_only()

            showing_stimulus = []
            # prepare showing datas
            if stimulus.is_count_correct:
                # count of correct unit of sequence
                correct_seq = []
                for i in range(0, len(stimulus.texts)):
                    if i < stimulus.unit_correct_count:
                        correct_seq += "*"
                    else:
                        correct_seq += stimulus.texts[i]

                # showing stimulus
                if stimulus.is_display_unit_correct == True:
                    showing_stimulus = correct_seq
                else:
                    showing_stimulus = stimulus.texts
            else:
                showing_stimulus = stimulus.texts

            # display
            if stimulus.is_count_correct:
                # Displaying count of correct sequence
                count_correct = self.visual.TextStim(win=self.win,
                                                     text="*" * stimulus.correct_count,
                                                     height=0.1,
                                                     bold=True,
                                                     colorSpace="rgb",
                                                     color=stimulus.color,
                                                     pos=(0, -0.3),
                                                     alignHoriz="center"
                                                     )
                count_correct.draw()

            text = self.visual.TextStim(win=self.win,
                                        text=" - ".join(showing_stimulus),
                                        height=stimulus.text_height,
                                        bold=True,
                                        colorSpace="rgb",
                                        color=stimulus.color)
            text.draw()

            # callback
            if stimulus.showing_time != 0:
                self.call_stimulus_showing_handler("seq texts", showing_stimulus, stimulus.showing_time)

            self.win.flip()
            print(str.format("showing stimulus: {0}, showing time: {1}sec", stimulus.texts, stimulus.showing_time))

    def show_stimuluses_with_step_counting(self, stimuluses, end_process = None):
        self.total_stimuluses = stimuluses
        self.show_stimuluses(stimuluses, end_process)

    def show_stimuluses(self, stimuluses, end_process = None):
        # save current step
        stimulus_length = len(stimuluses)
        if self.total_stimuluses != None:
            self.current_step = len(self.total_stimuluses) - stimulus_length

        stimulus = stimuluses[0]
        self.show_stimulus(stimulus)

        def show_next_stim():
            # Display manager needs main thread but interface manager occupies main thread when while while loop is running
            # so, display manager gets interface manager to interrupt for displaying stimulus
            self.intermediater.insert_interrupt(lambda: self.show_stimuluses(stimuluses=stimuluses[1:],
                                                                             end_process=end_process))

        if stimulus_length > 1:
            if stimulus.showing_time > 0:
                self.show_delay_after(stimulus.showing_time, show_next_stim)
            else:
                show_next_stim()
        else:
            # when stimulus list is over, display manager shows ending message
            def ending():
                def last_operation():
                    if end_process != None:
                        end_process()

                self.intermediater.insert_interrupt(last_operation)

            self.show_delay_after(stimulus.showing_time, ending)

    def show_text_bundle(self, bundle, end_process = None):
        if isinstance(bundle, Text_st_bundle):
            self.show_stimuluses(stimuluses=self.make_stimulus_in_text_bundle(bundle),
                                 end_process=end_process)

    def show_sequence_bundle(self, bundle, end_process = None):
        if isinstance(bundle, Sequence_st_bundle):
            self.show_stimuluses(stimuluses=self.make_stimulus_in_seq_bundle(Sequence_st_bundle),
                                 end_process=end_process)

    def show_package(self, pkg, end_process = None):
        """
        :param pkg: St_Package
        :param end_process: function, it called when last stimulus is ended
        """
        if isinstance(pkg, St_Package):
            self.show_stimuluses(stimuluses=self.make_stimulus_in_pkg(pkg),
                                 end_process=end_process)

    def show_packages(self, pkgs, end_process = None):
        """
        :param pkgs: list of St_Package
        :param end_process: function, it called when last stimulus of last package is ended
        """
        stimuluses = []
        for pkg in pkgs:
            if isinstance(pkg, St_Package):
                stimuluses += self.make_stimulus_in_pkg(pkg)
            elif isinstance(pkg, Text_st_unit):
                stimuluses += [pkg]
        self.show_stimuluses(stimuluses=stimuluses,
                             end_process=end_process)

    def show_packages_with_step_counting(self, pkgs, end_process = None):
        """
        The difference about show_packages is that this function calculates stimulus's step for matching with response
        (when the pair of stimulus-response is ("ABC", "1") then we have to know when the stimulus and response is occured, so I checked the event by step)

        :param pkgs: list of St_Package
        :param end_process: function, it called when last stimulus of last package is ended
        """
        stimuluses = []
        for pkg in pkgs:
            if isinstance(pkg, St_Package):
                stimuluses += self.make_stimulus_in_pkg(pkg)
            elif isinstance(pkg, Text_st_unit):
                stimuluses += [pkg]
        self.total_stimuluses = stimuluses
        self.show_stimuluses(stimuluses=stimuluses,
                             end_process=end_process)

    def show_stimulus_with_exception(self, stimulus):
        """
        This function is used to change display while current stimulus information is preserved (Non changing current_showing_stimlus)

        but, Do not use this function frequently, because this function causes confusing of source
        """
        self.visual_stimuli(stimulus)

    def make_stimulus_in_text_bundle(self, bundle):
        """
        This function unpacks bundle to make corresponding stimuluses

        :param bundle: St_bundle
        """
        stimuluses = []
        if isinstance(bundle, Text_st_bundle):
            for j in range(0, len(bundle.units)):
                stimuluses.append(bundle.units[j])
                if bundle.ISI_units[j].showing_time > 0:
                    stimuluses.append(bundle.ISI_units[j])
        return stimuluses

    def make_stimulus_in_seq_bundle(self, bundle):
        """
        This function unpacks sequence bundle to make corresponding stimuluses

        :param bundle: Sequence_st_bundle
        """
        stimuluses = []
        for k in range(0, len(bundle.units)):
            stimuluses.append(bundle.units[k])
            if bundle.ISI_units[k].showing_time > 0:
                stimuluses.append(bundle.ISI_units[k])
        return stimuluses

    def make_stimulus_in_pkg(self, pkg):
        """
        This function unpacks Package to make corresponding stimuluses

        :param bundle: St_Package
        """

        stimuluses = []
        if pkg.start_wait_seconds > 0:
            stimuluses.append(pkg.start_unit)

        for i in range(0, len(pkg.bundles)):
            bundle = pkg.bundles[i]
            if isinstance(bundle, Text_st_bundle):
                stimuluses += self.make_stimulus_in_text_bundle(bundle)
            elif isinstance(bundle, Sequence_st_bundle):
                stimuluses += self.make_stimulus_in_seq_bundle(bundle)

            if len(pkg.interval_units) > 0:
                if i != len(pkg.bundles)-1: # 맨 마지막 번들 뒤에는 번들 인터벌 넣어주지 않음
                    stimuluses.append(pkg.interval_units[i])
        return stimuluses

    def close_window(self):
        print("closed window")
        core.quit()
        self.win.close()

    def show_delay_after(self, seconds, process):
        self.timer.start(seconds, process)

    def show_next_directly(self):
        self.timer.direct_proc()

class Input_interface_manager:
    device_keyboard = "keyboard"
    device_mouse = "mouse"

    def __init__(self,
                 start_keys,
                 stop_keys,
                 intermediater,
                 mouse_event_pooling_interval = None,
                 devices="keyboard"):
        self.start_keys = start_keys
        self.stop_keys = stop_keys
        self.device_names = devices.split("|")
        self.interrupt_queue = queue.Queue()
        self.stop_proc = None
        self.is_stop_monitoring = False
        self.mouseListener = None
        self.last_mouse_event_time = 0.0
        self.mouse_event_pooling_interval = mouse_event_pooling_interval

        if isinstance(intermediater, Intermediater):
            self.intermediater = intermediater
        else:
            raise Exception("intermediater Type Error!")

    def set_stop_process(self, stop_proc):
        self.stop_proc = stop_proc

    def wait_start(self):
        if Input_interface_manager.device_keyboard in self.device_names:
            psychopy.event.clearEvents()
            while True:
                keys = psychopy.event.getKeys()
                if self.is_inputted_sth(keys):
                    input = self.get_input(keys)
                    print("Input: ", input)
                    if input in self.start_keys:
                        print("Start!")
                        break
                    elif input in self.stop_keys:
                        if self.stop_proc != None:
                            self.stop_proc()

    def insert_interrupt(self, is_interrupt):
        self.interrupt_queue.put(is_interrupt)

    def set_interrupt_operation(self, doSomething):
        self.interrupt_operation = doSomething

    def monitoring(self):
        while True:
            if self.interrupt_queue.empty() == False and self.interrupt_queue.get() == True:
                # if something is interrupted, It processes the thing first
                # This code is needed because The Keyboard listening is busy-wait so We can't anything while listening is processed
                if self.interrupt_operation != None:
                    self.interrupt_operation()
                    # Interrupt가 일어난 경우에, stop monitoring 콜이 발생한 경우, 루프 중지
                    if self.is_stop_monitoring:
                        self.is_stop_monitoring = False
                        break

            if Input_interface_manager.device_keyboard in self.device_names:
                keys = psychopy.event.getKeys()
                if self.is_inputted_sth(keys):
                    input = self.get_input(keys)
                    print("Input: ", input)
                    if input in self.stop_keys:
                        print("Monitoring is Stopped")
                        if self.stop_proc != None:
                            self.stop_proc()
                        break
                    else:
                        if self.intermediater is not None:
                            self.intermediater.listen_key_input(input)
                        else:
                            print("Event Manager is not set out")

            if Input_interface_manager.device_mouse in self.device_names:
                if self.intermediater is not None and self.intermediater.is_use_event_manager():
                    if self.mouseListener == None:
                        self.mouseListener = psychopy.event.Mouse(visible=True)

                    if time.time() - self.last_mouse_event_time >= self.mouse_event_pooling_interval:
                        self.last_mouse_event_time = time.time()

                        width, height = self.intermediater.get_win_size()
                        relative_pos = self.mouseListener.getPos()

                        x = (width / 2) * relative_pos[0] + (width / 2)
                        y = ((-1) * height / 2) * relative_pos[1] + (height / 2)

                        self.intermediater.listen_mouse_input((x, y))
                else:
                    print("Event Manager is not set out")

    def is_inputted_sth(self, keys):
        return len(keys) != 0

    def get_input(self, keys):
        return keys[0]

    def set_is_stop_monitoring(self, is_stop_monitoring):
        self.is_stop_monitoring = is_stop_monitoring

class Event_manager:
    def __init__(self,
                 is_activate=True,
                 is_activate_one_key_input=False,
                 is_activate_multiple_key_input=False,
                 is_active_mouse_input = False,
                 valid_keys=None):
        self.valid_keys = valid_keys

        # Flag to activate
        self.is_activate = is_activate
        self.is_activate_one_key_input = is_activate_one_key_input
        self.is_activate_multiple_key_input = is_activate_multiple_key_input
        self.is_active_mouse_input = is_active_mouse_input

        # Check keys for activating multiple key events
        self.input_key_buffer = []
        self.target_input_count = 0

        # input handlers
        self.mouse_input_handler = None
        self.single_key_input_handler = None
        self.multiple_key_input_handler = None

    """
    Listening
    """
    def listen_mouse_input(self, mousePos, current_stimulus):
        """
        :param mousePos: mouse position
        :param current_stimulus: current stimulus when event is occurred
        """
        if self.is_activate == True and self.is_active_mouse_input == True:
            self.mouse_input_handler(mousePos, current_stimulus)

    def listen_key_input(self, input, current_stimulus):
        """
        Processing Key input

        :param input: key input(string)
        :param current_stimulus: current stimulus when event is occurred
        """

        if self.valid_keys == None:
            return

        print("active: ", self.is_activate)
        if self.is_activate == True and (input in self.valid_keys):
            self.input_key_buffer.append(input)

            self.listen_one_key_input(input, current_stimulus)
            self.listen_multiple_key_input(self.input_key_buffer , current_stimulus)

    def listen_one_key_input(self, input, current_stimulus):
        """
        Processing Single Key input

        :param input: key input(string)
        :param current_stimulus: current stimulus when event is occurred
        """
        if self.is_activate_one_input == True:
            self.single_key_input_handler(input, current_stimulus)

    def listen_multiple_key_input(self, inputs, current_stimulus):
        """
        Processing Multiple Key input

        :param input: key input(string)
        :param current_stimulus: current stimulus when event is occurred
        """
        if self.is_activate_multiple_input == True:
            if len(self.input_key_buffer) == self.target_input_count:
                self.multiple_key_input_handler(inputs, current_stimulus)
                self.input_key_buffer = []
        else:
            self.input_key_buffer = [] # if not use, clean input buffer

    """
    Handler Setting
    """
    def set_mouse_input_handler(self, function):
        """
        Setting mouse input hanlder

        :param function: mouse input hanlder
        """
        self.mouse_input_handler = function

    def set_single_key_input_handler(self, function):
        """
        Setting key single input hanlder

        :param function: single key input hanlder
        """
        self.single_key_input_handler = function

    def set_multiple_key_input_handler(self, function):
        """
        Setting key multiple input hanlder

        :param function: multiple key input hanlder
        """
        self.multiple_key_input_handler = function

    """
    Activation Setting
    """
    def set_is_activate(self, is_activate):
        """
        :param is_activate: Flag to activate event manager(boolean)
        """
        self.is_activate = is_activate

    def set_is_active_mouse_input(self, is_active):
        self.is_active_mouse_input = is_active

    def set_is_activate_one_input(self, is_one_activate):
        """
        :param is_one_activate: Flag to activate single key event(boolean)
        """
        self.is_activate_one_input = is_one_activate

    def set_is_activate_multiple_input(self, is_multiple_activate, target_input_count):
        """
        :param is_multiple_activate: Flag to activate multiple key event(boolean)
        :param target_input_count: count of buffer keys when occurring event
        """
        self.input_key_buffer = []
        self.is_activate_multiple_input = is_multiple_activate
        self.target_input_count = target_input_count

class Experiment:
    def __init__(self,
                 monitor_size,
                 is_full_screen,
                 data_dir_path,
                 participant_name,
                 ready_keys = [],
                 start_keys = [],
                 stop_keys = [],
                 valid_keys = None,
                 input_device = "keyboard"):
        self.previous_input = "" # fMRI 시퀀스 기기 특성상 계속 입력되는것 방지

        """
        Setting Data
        """
        self.data_dir_path = data_dir_path
        self.participant_name = participant_name
        self.ready_keys = ready_keys

        """
        Display and interface Setting
        """
        self.intermediater = Intermediater()

        self.event_manager = Event_manager(valid_keys=valid_keys)
        self.interface = Input_interface_manager(start_keys=start_keys,
                                                 stop_keys=stop_keys,
                                                 intermediater=self.intermediater,
                                                 devices=input_device)

        self.display_manager = Psy_display_manager(intermediater=self.intermediater)
        self.display_manager.open_window(size=monitor_size, color=[-1, -1, -1], is_full_screen=is_full_screen)

        self.intermediater.set_input_interface_manager(self.interface)
        self.intermediater.set_display_manager(self.display_manager)
        self.intermediater.set_event_manager(self.event_manager)

        self.start_time = time.time()

        def log_showing(type, stimulus, showing_time):
            # ["Step", "Event_Type", "Stimulus", "display_seconds", "start_seconds"]
            self.stimulus_csv_manager.write_row([self.display_manager.current_step, type, stimulus, showing_time, time.time() - self.start_time])
            self.previous_input = "" # 다음 화면 넘어갔을때, log 기록 되도록함

        self.display_manager.set_stimulus_showing_handler(log_showing)

        self.interface.set_stop_process(lambda: self.display_manager.close_window())

        """
        Setting Events
        """
        def log_response(response):
            # ["Step", "Response", "seconds"]
            self.response_csv_manager.write_row([self.display_manager.current_step, response, time.time() - self.start_time])
            print("response logging... step: ", self.display_manager.current_step, "response: ", response, "time: ", time.time() - self.start_time)

        def single_key_input_handler(input, current_stimulus):
            print("proc single input", input)

            if isinstance(current_stimulus, Sequence_st_text_unit):
                if current_stimulus.is_count_correct:
                    count_coding = current_stimulus.unit_correct_count

                    if self.previous_input != input: # replay 실험후 지워야
                        log_response(input)
                    self.previous_input = input  # replay 실험후 지워야

                    # show coded stimuli
                    if current_stimulus.texts[count_coding] == input:
                        if count_coding < len(current_stimulus.texts) - 1:
                            # 그냥 그대로 보여줌
                            unit = Sequence_st_text_unit(text_units=current_stimulus.text_units,
                                                        color=current_stimulus.color,
                                                        text_height=current_stimulus.text_height,
                                                        is_count_correct=True,
                                                        unit_correct_count=count_coding+1,
                                                        correct_count=current_stimulus.correct_count)
                        elif count_coding == len(current_stimulus.texts) - 1:
                            # 별 표시 보여줌
                            unit = Sequence_st_text_unit(text_units=current_stimulus.text_units,
                                                         color=current_stimulus.color,
                                                         text_height=current_stimulus.text_height,
                                                         is_count_correct=True,
                                                         unit_correct_count=0,
                                                         correct_count=current_stimulus.correct_count+1)
                            log_response(current_stimulus.text_units) # To record all matched event
                            self.previous_input = "" # Clear(다음 시퀀스 입력 로그 찍을수 있도록)
                        self.display_manager.show_stimulus(unit)
                else:
                    log_response(input)

        def multiple_key_input_handler(inputs, current_stimulus):
            print("proc multiple_input", inputs)
            log_response(inputs)

            if isinstance(self.display_manager.current_showing_stimlus, Sequence_st_text_unit):
                self.display_manager.show_next_directly()

        self.event_manager.set_single_key_input_handler(single_key_input_handler)
        # self.event_manager.set_multiple_key_input_handler(multiple_key_input_handler)

    def setting_log(self, iteration):
        self.stimulus_csv_manager = CsvManager(dir_path=self.data_dir_path,
                                               file_name="stimulus_" + self.participant_name + "_" + "r" + str(iteration + 1)) # start from 1
        self.stimulus_csv_manager.write_header(["Step", "Event_Type", "Stimulus", "display_seconds", "start_seconds"])
        self.response_csv_manager = CsvManager(dir_path=self.data_dir_path,
                                               file_name="response_" + self.participant_name + "_" + "r" + str(iteration + 1))
        self.response_csv_manager.write_header(["Step", "Response", "seconds"])

    def wait_pkg(self, pkgs, iteration, addition_end_proc = None):
        self.setting_log(iteration)

        return_value = self.display_manager.wait_start(iteration=iteration, ready_keys=self.ready_keys, stop_keys=self.interface.stop_keys)

        def start():
            self.start_time = time.time()

            def end_proc():
                self.invalid_input_event()
                if addition_end_proc != None:
                    addition_end_proc()

            self.display_manager.show_packages_with_step_counting(pkgs=pkgs,
                                                                  end_process=end_proc)
            self.interface.monitoring()
        if return_value == True:
            start()

    def wait_blocks(self, blocks, iteration):
        """
        :param blocks: list of package
        :param iteration: start block index
        """
        def end():
            if iteration+1 < len(blocks):
                # 기존꺼 멈추고
                self.interface.set_is_stop_monitoring(True)
                # 새로 실행
                self.wait_blocks(blocks, iteration+1)
            else:
                self.display_manager.show_stimulus(Text_st_unit("+", showing_time=0))

        self.wait_pkg(pkgs=blocks[iteration], iteration=iteration, addition_end_proc=end)

    def wait_stimuluses(self, stimuluses, iteration, addition_end_proc = None):
        self.setting_log(iteration)

        return_value = self.display_manager.wait_start(iteration=iteration,
                                                       ready_keys=self.ready_keys,
                                                       stop_keys=self.interface.stop_keys)

        def start():
            self.start_time = time.time()

            def end_proc():
                self.invalid_input_event()
                if addition_end_proc != None:
                    addition_end_proc()

            self.display_manager.show_stimuluses_with_step_counting(stimuluses=stimuluses,
                                                                    end_process=end_proc)
            self.interface.monitoring()
        if return_value == True:
            start()

    def invalid_input_event(self):
        self.event_manager.set_is_activate_one_input(False)
        self.event_manager.set_is_activate_multiple_input(False, 0)
        print("invalid event_manager")
        self.stimulus_csv_manager.write_row([-1, "End", "End", 0, time.time() - self.start_time])

if __name__ == "__main__":
    """
    Basic Usage
    """
    intermediater = Intermediater()

    event_manager = Event_manager()
    interface = Input_interface_manager(start_keys=["s"], stop_keys=["q"], intermediater=intermediater, devices="keyboard")
    p = Psy_display_manager(intermediater=intermediater)

    intermediater.set_event_manager(event_manager)
    intermediater.set_input_interface_manager(interface)
    intermediater.set_display_manager(p)

    p.open_window([200, 200], [-1, -1, -1])

    p.show_stimuluses([Text_st_unit("1", showing_time=2), Text_st_unit("2", showing_time=3)])

    interface.monitoring()
    p.close_window()

    """
    Experiment Usage
    """
    source_path = "/Users/yoonseojin/Statistics_sj/CLMN/Replay_Exp"

    data_dir_path = source_path
    participant_name = "seojin"

    exp = Experiment(monitor_size=[400, 400],
                     is_full_screen=False,
                     data_dir_path=data_dir_path,
                     participant_name=participant_name,
                     ready_keys=["r"],
                     start_keys=["s"],
                     stop_keys=["q"],
                     input_device="keyboard")
    exp.wait_stimuluses(stimuluses=[Text_st_unit("ABC", showing_time=1), Text_st_unit("DEF", showing_time=1)],
                        iteration=0)

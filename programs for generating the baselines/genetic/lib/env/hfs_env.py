#############################################################################################################################################################################################################
### PRINTED CIRCUIT BOARDS PRODUCTION - A TWO-STAGE HYBRID FLOWSHOP SCHEDULING PROBLEM ######################################################################################################################
#############################################################################################################################################################################################################
### Author: Sebastian Lang (sebastian.lang@ovgu.de) #########################################################################################################################################################
#############################################################################################################################################################################################################

### Libraries ###############################################################################################################################################################################################

from lib.env import schedulers
import salabim as sim
import numpy as np
import pandas as pd
import math
import random

#############################################################################################################################################################################################################

### Global Variables ########################################################################################################################################################################################

# Visualization
ENV_W = 1000
ENV_H = 600
REF_WIDTH = 110
GLOBAL_QUEUE_WIDTH = REF_WIDTH
GLOBAL_PROCESS_WIDTH = REF_WIDTH
REF_HEIGHT = 60
GLOBAL_SOURCE_DRAIN_RADIUS = REF_HEIGHT / 2
GLOBAL_QUEUE_HEIGHT = REF_HEIGHT
GLOBAL_PROCESS_HEIGHT = REF_HEIGHT
GLOBAL_FONTSIZE = 16
X_0 = 50
Y_0 = 300
Y_GLOBAL_SOURCE_DRAIN = Y_0 + GLOBAL_SOURCE_DRAIN_RADIUS
Y_GLOBAL_BUFFER = Y_0 + ((REF_HEIGHT - GLOBAL_QUEUE_HEIGHT) / 2)
Y_GLOBAL_PROCESS = Y_0 + ((REF_HEIGHT - GLOBAL_PROCESS_HEIGHT) / 2)

#############################################################################################################################################################################################################

### Modeling Objects ########################################################################################################################################################################################

class Job(sim.Component):
    def setup(
        self,
        hfs_env,
        job_id,
        due_date,
        family,
        t_smd,
        t_aoi,
        scaled_due_date,
        scaled_family,
        scaled_t_smd,
        scaled_t_aoi,
    ):

        # hybrid flow shop environment
        self.hfs_env = hfs_env
        # initial attributes
        self.job_id = job_id
        self.due_date = due_date
        self.family = family
        self.t_smd = t_smd
        self.t_aoi = t_aoi
        self.scaled_due_date = scaled_due_date
        self.scaled_family = scaled_family
        self.scaled_t_smd = scaled_t_smd
        self.scaled_t_aoi = scaled_t_aoi
        # visualization
        if self.hfs_env.env.animate() == True:
            self.draw_animation(x=X_0, y=Y_GLOBAL_SOURCE_DRAIN)
        
    def process(self):

        # enter SMD stage
        self.enter(self.hfs_env.global_smd_queue)
        if self.hfs_env.env.animate() == True:
            self.img.remove()
            self.hfs_env.global_smd_queue.update_info_text()
        yield self.passivate()

        # enter AOI stage
        self.enter(self.hfs_env.global_aoi_queue)
        if self.hfs_env.passive_aois:
            aoi = self.hfs_env.passive_aois[0]
            self.hfs_env.passive_aois.remove(aoi)
            self.hfs_env.active_aois.append(aoi)
            aoi.activate()
        if self.hfs_env.env.animate() == True:
            self.img.remove()
            self.hfs_env.global_aoi_queue.update_info_text()
        yield self.passivate()

        # calculate objectives
        self.hfs_env.jobs_processed += 1
        if self.hfs_env.env.now() > self.due_date:
            self.hfs_env.total_tardiness += self.hfs_env.env.now() - self.due_date
            if self.hfs_env.env.animate() == True:
                self.hfs_env.info_tardiness.text = "Total Tardiness: " + str(
                    self.hfs_env.total_tardiness
                )
        if self.hfs_env.env.animate() == True:
            self.draw_animation(
                x=self.hfs_env.drain.img[0].x, y=self.hfs_env.drain.img[0].y
            )
            self.hfs_env.info_num_jobs.text = "Completed Jobs: {}".format(
                self.hfs_env.jobs_processed
            )
            yield self.hold(0)
        if self.hfs_env.jobs_processed == self.hfs_env.num_jobs:
            self.hfs_env.state = "end of simulation"
            self.hfs_env.makespan = self.hfs_env.env.now()
            if self.hfs_env.env.animate() == True:
                self.hfs_env.info_makespan.text = "Makespan: " + str(
                    self.hfs_env.makespan
                )
                if self.hfs_env.freeze_window_at_endsim:
                    self.hfs_env.env.an_menu()
            yield self.hold(0)

        # destroy job
        del self

    def draw_animation(self, x, y):

        self.img = sim.AnimateCircle(
            radius=10,
            x=x,
            y=y,
            fillcolor="limegreen",
            linecolor="black",
            text=str(self.job_id),
            fontsize=15,
            textcolor="black",
            parent=self,
            screen_coordinates=True,
        )


class Source(sim.Component):
    def setup(
        self,
        hfs_env,
        img_w=GLOBAL_SOURCE_DRAIN_RADIUS,
        img_h=GLOBAL_SOURCE_DRAIN_RADIUS,
        img_x=X_0,
        img_y=Y_GLOBAL_SOURCE_DRAIN,
    ):

        # hybrid flow shop environment
        self.hfs_env = hfs_env

        # visualization
        if self.hfs_env.env.animate() == True:
            self.img_w = img_w
            self.img_h = img_h
            self.img_x = img_x
            self.img_y = img_y
            self.img = [
                sim.AnimateCircle(
                    radius=img_w,
                    x=img_x,
                    y=img_y,
                    fillcolor="white",
                    linecolor="black",
                    linewidth=2,
                    layer=2,
                    arg=(img_x + img_w, img_y + img_h),
                    screen_coordinates=True,
                ),
                sim.AnimateCircle(
                    radius=0.3 * img_w,
                    x=img_x,
                    y=img_y,
                    fillcolor="black",
                    linecolor="black",
                    layer=1,
                    screen_coordinates=True,
                ),
            ]

    def process(self):
       
        # generate jobs
        for job in self.hfs_env.sequence:

            Job(
                hfs_env=self.hfs_env,
                job_id=job["id"],
                due_date=job["due date"],
                family=job["family"],
                t_smd=job["t_smd"],
                t_aoi=job["t_aoi"],
                scaled_due_date=job["scaled due date"],
                scaled_family=job["scaled family"],
                scaled_t_smd=job["scaled t_smd"],
                scaled_t_aoi=job["scaled t_aoi"],
            )
         
            yield self.hold(0)

        for smd in self.hfs_env.smds:
            smd.activate()

        # step-by-step mode
        if self.hfs_env.freeze_window_after_initsim:
            self.hfs_env.env.an_menu()


class Queue(sim.Queue):
    def setup(
        self,
        hfs_env,
        predecessors,
        img_w=None,
        img_h=None,
        img_x=None,
        img_y=None,
        img_slots=None,
    ):

        # hybrid flow shop environment
        self.hfs_env = hfs_env

        # predecessors
        self.predecessors = predecessors

        # parameters for visualization
        self.img_w = img_w
        self.img_h = img_h
        self.img_x = img_x
        self.img_y = img_y

        # visualization
        if self.hfs_env.env.animate() == True:
            self.img = sim.AnimateRectangle(
                spec=(0, 0, self.img_w, self.img_h),
                x=self.img_x,
                y=self.img_y,
                fillcolor="white",
                linecolor="black",
                linewidth=2,
                layer=1,
                arg=(self.img_x + (self.img_w / 2), self.img_y + (self.img_h / 2)),
                screen_coordinates=True,
            )
            self.predecessor_connections = [
                sim.AnimateLine(
                    spec=(
                        predecessor.img_x + predecessor.img_w,
                        predecessor.img_y
                        if predecessor.__class__.__name__ == "Source"
                        else (predecessor.img_y + predecessor.img_h / 2),
                        self.img_x,
                        self.img_y + self.img_h / 2,
                    ),
                    linecolor="black",
                    linewidth=2,
                    layer=2,
                    screen_coordinates=True,
                )
                if self.predecessors[0].__class__.__name__ != "Queue"
                else (
                    [
                        sim.AnimateLine(
                            spec=(
                                predecessor.img_x + predecessor.img_w,
                                predecessor.img_y + predecessor.img_h / 2,
                                predecessor.img_x + predecessor.img_w + 10,
                                predecessor.img_y + predecessor.img_h / 2,
                            ),
                            linecolor="black",
                            linewidth=2,
                            layer=2,
                            screen_coordinates=True,
                        ),
                        sim.AnimateLine(
                            spec=(
                                predecessor.img_x + predecessor.img_w + 10,
                                predecessor.img_y + predecessor.img_h / 2,
                                predecessor.img_x + predecessor.img_w + 10,
                                self.img_y + self.img_h / 2,
                            ),
                            linecolor="black",
                            linewidth=2,
                            layer=2,
                            screen_coordinates=True,
                        ),
                        sim.AnimateLine(
                            spec=(
                                predecessor.img_x + predecessor.img_w + 10,
                                self.img_y + self.img_h / 2,
                                self.img_x,
                                self.img_y + self.img_h / 2,
                            ),
                            linecolor="black",
                            linewidth=2,
                            layer=2,
                            screen_coordinates=True,
                        ),
                    ]
                )
                for predecessor in self.predecessors
            ]
            self.info = sim.AnimateText(
                text="# jobs: 0",
                x=self.img.x + 8,
                y=self.img.arg[1] - 8,
                fontsize=GLOBAL_FONTSIZE,
                textcolor="black",
                screen_coordinates=True,
            )

    def update_info_text(self):
        self.info.text = "# jobs: {}".format(len(self))


class SMD(sim.Component):
    def setup(
        self,
        hfs_env,
        global_queue,
        local_queue,
        img_x=None,
        img_y=None,
        img_w=GLOBAL_PROCESS_WIDTH,
        img_h=GLOBAL_PROCESS_HEIGHT,
        info_x=None,
        info_y=None,
    ):

        # hybrid flow shop environment
        self.hfs_env = hfs_env

        # predecessors
        self.global_queue = global_queue
        self.local_queue = local_queue

        # parameters for visualization
        self.img_x = img_x
        self.img_y = img_y
        self.img_w = img_w
        self.img_h = img_h
        self.info_x = info_x
        self.info_y = info_y

        # state variables
        self.job = None
        self.setuptype = 0
        self.scaled_setuptype = 0
        self.setup_to = 0
        self.state = "idle"

        # visualization
        if hfs_env.env.animate() == True:
            self.img = sim.AnimatePolygon(
                spec=(
                    0,
                    0,
                    self.img_w - (self.img_w / 300) * 50,
                    0,
                    self.img_w,
                    self.img_h / 2,
                    self.img_w - (self.img_w / 300) * 50,
                    self.img_h,
                    0,
                    self.img_h,
                    0,
                    0,
                ),
                x=self.img_x,
                y=self.img_y,
                fillcolor="white",
                linecolor="black",
                linewidth=2,
                text=self.name() + "\n\nsetuptype = 0\nidle",
                fontsize=GLOBAL_FONTSIZE - 1,
                textcolor="black",
                layer=1,
                screen_coordinates=True,
            )
            self.predecessor_connections = [
                [
                    sim.AnimateLine(
                        spec=(
                            self.global_queue.img_x + self.global_queue.img_w,
                            self.global_queue.img_y + self.global_queue.img_h / 2,
                            self.global_queue.img_x + self.global_queue.img_w + 10,
                            self.global_queue.img_y + self.global_queue.img_h / 2,
                        ),
                        linecolor="black",
                        linewidth=2,
                        layer=2,
                        screen_coordinates=True,
                    ),
                    sim.AnimateLine(
                        spec=(
                            self.global_queue.img_x + self.global_queue.img_w + 10,
                            self.global_queue.img_y + self.global_queue.img_h / 2,
                            self.global_queue.img_x + self.global_queue.img_w + 10,
                            self.img_y + 20,
                        ),
                        linecolor="black",
                        linewidth=2,
                        layer=2,
                        screen_coordinates=True,
                    ),
                    sim.AnimateLine(
                        spec=(
                            self.global_queue.img_x + self.global_queue.img_w + 10,
                            self.img_y + 20,
                            self.img_x,
                            self.img_y + 20,
                        ),
                        linecolor="black",
                        linewidth=2,
                        layer=2,
                        screen_coordinates=True,
                    ),
                ],
                sim.AnimateLine(
                    spec=(
                        self.local_queue.img_x + self.local_queue.img_w,
                        self.local_queue.img_y + self.local_queue.img_h / 2,
                        self.img_x,
                        self.local_queue.img_y + self.local_queue.img_h / 2,
                    ),
                    linecolor="black",
                    linewidth=2,
                    layer=2,
                    screen_coordinates=True,
                ),
            ]

    def process(self):

        while True:

            # idle state
            if not self.global_queue and not self.local_queue:
                self.state = "idle"
                if self.hfs_env.env.animate() == True:
                    self.set_status(status=self.state)
                yield self.passivate()

            # pick next job
            if self.local_queue:
                self.job = self.local_queue.pop()
                if self.hfs_env.env.animate() == True:
                    self.local_queue.update_info_text()
            else:
                if self.hfs_env.step_execution and self.hfs_env.smd_scheduling_model:
                    self.hfs_env.state = "smd scheduling"
                    self.hfs_env.observed_machine = self
                    if self.hfs_env.tracing:
                        print(
                            "\n--- waiting for job scheduling for {} ---\n".format(
                                self.name()
                            )
                        )
                    yield self.hold(0)
                    # the environment may stops and waits here to proceed, if run_by_step == True
                else:
                    if (
                        self.hfs_env.smd_scheduler.__class__.__base__.__name__
                        == "DRL_Scheduler"
                    ):
                        transition = self.hfs_env.smd_scheduler.select_job(self)
                        self.hfs_env.num_forward += len(transition["states"])
                    else:
                        self.hfs_env.smd_scheduler.select_job(self)

            if self.job != None:

                if self.hfs_env.env.animate() == True:
                    self.hfs_env.global_smd_queue.update_info_text()
                    self.job.draw_animation(
                        x=self.img_x + (self.img_w / 2), y=self.img_y + (self.img_h / 2)
                    )

                # setup state
                if self.setuptype == self.job.family:
                    self.state = "minor setup"
                    if self.hfs_env.env.animate() == True:
                        self.set_status(status=self.state)
                    yield self.hold(20)
                else:
                    self.state = "major setup"
                    self.hfs_env.num_major_setups += 1
                    if self.hfs_env.env.animate() == True:
                        self.set_status(status=self.state)
                        self.hfs_env.info_setups.text = "Major Setups: " + str(
                            self.hfs_env.num_major_setups
                        )
                    self.setuptype = 0
                    self.setup_to = self.job.family
                    yield self.hold(65)
                    self.setuptype = self.job.family
                    self.scaled_setuptype = self.job.scaled_family
                    self.setup_to = 0

                # active state
                self.state = "active"
                if self.hfs_env.env.animate() == True:
                    self.set_status(status=self.state)
                yield self.hold(self.job.t_smd)
                self.job.activate()
                self.job = None

    def set_status(self, status):

        dict_status = {
            "idle": "white",
            "active": "lime",
            "minor setup": "yellow",
            "major setup": "tomato",
        }

        self.img.fillcolor = dict_status.get(status)
        self.img.text = (
            self.name() + "\n\nsetuptype = " + str(self.setuptype) + "\n" + status
        )

    def calc_workload(self):

        if self.state == "active":
            workload = self.remaining_duration()
        elif self.state in ["minor setup", "major setup"]:
            workload = self.remaining_duration() + self.job.t_smd
        else:
            workload = 0

        for job in self.local_queue:
            workload += 20 + job.t_smd

        return workload


class AOI(sim.Component):
    def setup(
        self,
        hfs_env,
        global_queue,
        img_x=None,
        img_y=None,
        img_w=GLOBAL_PROCESS_WIDTH,
        img_h=GLOBAL_PROCESS_HEIGHT,
    ):

        # hybrid flow shop environment
        self.hfs_env = hfs_env

        # predecessor
        self.global_queue = global_queue

        # parameters for visualization
        self.img_x = img_x
        self.img_y = img_y
        self.img_w = img_w
        self.img_h = img_h

        # state variables
        self.job = None
        self.state = "idle"

        # visualization
        if self.hfs_env.env.animate() == True:
            self.img = sim.AnimatePolygon(
                spec=(
                    0,
                    0,
                    self.img_w - (self.img_w / 300) * 50,
                    0,
                    self.img_w,
                    self.img_h / 2,
                    self.img_w - (self.img_w / 300) * 50,
                    self.img_h,
                    0,
                    self.img_h,
                    0,
                    0,
                ),
                x=self.img_x,
                y=self.img_y,
                fillcolor="white",
                linecolor="black",
                linewidth=2,
                text=self.name() + "\n\nidle",
                fontsize=GLOBAL_FONTSIZE,
                textcolor="black",
                layer=1,
                screen_coordinates=True,
            )
            self.predecessor_connection = sim.AnimateLine(
                spec=(
                    self.global_queue.img_x + self.global_queue.img_w,
                    self.global_queue.img_y + self.global_queue.img_h / 2,
                    self.img_x,
                    self.img_y + self.img_h / 2,
                ),
                linecolor="black",
                linewidth=2,
                layer=2,
                screen_coordinates=True,
            )

    def process(self):

        while True:

            # idle state
            if not self.global_queue:
                self.state = "idle"
                if self.hfs_env.env.animate() == True:
                    self.set_status(status=self.state)
                self.hfs_env.active_aois.remove(self)
                self.hfs_env.passive_aois.append(self)
                yield self.passivate()

            # pick next job
            if self.hfs_env.step_execution and self.hfs_env.aoi_scheduling_model:
                self.hfs_env.state = "aoi scheduling"
                self.hfs_env.observed_machine = self
                if self.hfs_env.tracing:
                    print(
                        "\n--- waiting for job scheduling for {} ---\n".format(
                            self.name()
                        )
                    )
                yield self.hold(0)
                # the environment may stops and waits here to proceed, if run_by_step == True
            else:
                if (
                    self.hfs_env.aoi_scheduler.__class__.__base__.__name__
                    == "DRL_Scheduler"
                ):
                    transition = self.hfs_env.aoi_scheduler.select_job(self)
                    self.hfs_env.num_forward += len(transition["states"])
                else:
                    self.hfs_env.aoi_scheduler.select_job(self)

            if self.hfs_env.env.animate() == True:
                self.hfs_env.global_aoi_queue.update_info_text()
                self.job.draw_animation(
                    x=self.img_x + (self.img_w / 2), y=self.img_y + (self.img_h / 2)
                )

            # setup state
            self.state = "setting-up"
            if self.hfs_env.env.animate() == True:
                self.set_status(status=self.state)
            yield self.hold(25)

            # active state
            self.state = "active"
            if self.hfs_env.env.animate() == True:
                self.set_status(status=self.state)
            yield self.hold(self.job.t_aoi)
            self.job.activate()
            self.job = None

    def set_status(self, status):

        dict_status = {"idle": "white", "active": "lime", "setting-up": "yellow"}

        self.img.fillcolor = dict_status.get(status)
        self.img.text = self.name() + "\n\n" + status

    def calc_workload(self):

        if self.state == "active":
            workload = self.remaining_duration()
        elif self.state == "setting-up":
            workload = self.remaining_duration() + self.job.t_aoi
        else:
            workload = 0

        return workload


class Drain:
    def __init__(
        self,
        hfs_env,
        predecessors,
        img_x=None,
        img_y=None,
        img_w=GLOBAL_SOURCE_DRAIN_RADIUS,
        img_h=GLOBAL_SOURCE_DRAIN_RADIUS,
    ):

        # hybrid flow shop environment
        self.hfs_env = hfs_env

        # predecessors
        self.predecessors = predecessors

        # parameters for visualization
        self.img_x = img_x
        self.img_y = img_y
        self.img_w = img_w
        self.img_h = img_h

        # visualization
        self.img = [
            sim.AnimateCircle(
                radius=img_w,
                x=self.img_x,
                y=self.img_y,
                fillcolor="white",
                linecolor="black",
                linewidth=2,
                layer=1,
                screen_coordinates=True,
            ),
            sim.AnimateLine(
                spec=(
                    img_w * math.cos(math.radians(45)) * (-1),
                    img_w * math.sin(math.radians(45)) * (-1),
                    img_w * math.cos(math.radians(45)),
                    img_w * math.sin(math.radians(45)),
                ),
                x=self.img_x,
                y=self.img_y,
                linecolor="black",
                linewidth=2,
                layer=1,
                arg=(self.img_x + img_w, self.img_y + img_w),
                screen_coordinates=True,
            ),
            sim.AnimateLine(
                spec=(
                    img_w * math.cos(math.radians(45)) * (-1),
                    img_w * math.sin(math.radians(45)),
                    img_w * math.cos(math.radians(45)),
                    img_w * math.sin(math.radians(45)) * (-1),
                ),
                x=self.img_x,
                y=self.img_y,
                linecolor="black",
                linewidth=2,
                layer=1,
                screen_coordinates=True,
            ),
        ]
        self.predecessor_connections = [
            sim.AnimateLine(
                spec=(
                    predecessor.img_x + predecessor.img_w,
                    predecessor.img_y + predecessor.img_h / 2,
                    self.img_x - self.img_w,
                    self.img_y,
                ),
                linecolor="black",
                linewidth=2,
                layer=2,
                screen_coordinates=True,
            )
            for predecessor in self.predecessors
        ]


class GeneticScheduler(sim.Component):
    def setup(self, schedulers_list_smd, schedulers_list_aoi, embedded_schedulers_smd, embedded_schedulers_aoi, switches, hfs_env):
            # hybrid flow shop environment
            self.hfs_env = hfs_env

            # switching points of schedulers
            self.switches = switches

            # scheduler list
            self.schedulers_list_smd = schedulers_list_smd
            self.schedulers_list_aoi = schedulers_list_aoi

            #embedded schedulers
            self.embedded_schedulers_smd = embedded_schedulers_smd
            self.embedded_schedulers_aoi = embedded_schedulers_aoi
    
    def process(self):
        for i in range(len(self.schedulers_list_smd)):

            # switch the schedulers of hfs env
            self.hfs_env.smd_scheduler = self.embedded_schedulers_smd[self.schedulers_list_smd[i]]
            self.hfs_env.aoi_scheduler = self.embedded_schedulers_aoi[self.schedulers_list_aoi[i]]

            # wait period length
            period_length = self.switches[i]
            yield self.hold(period_length)

#############################################################################################################################################################################################################

### Discrete Event Simulation Environment ###################################################################################################################################################################

class HFS_Env:
    def __init__(
        self,
        problem,
        smd_scheduling_class=None,
        aoi_scheduling_class=None,
        smd_scheduling_model=None,
        aoi_scheduling_model=None,
        schedulers_list_smd=None,
        schedulers_list_aoi=None,
        switches=None,
        noise_model=None,
        step_execution=False,
        animation=False,
        freeze_window_after_initsim=False,
        freeze_window_at_endsim=False,
        tracing=False,
    ):

        # input data
        self.problem = problem

        # artificial neural networks
        self.smd_scheduling_model = smd_scheduling_model
        self.aoi_scheduling_model = aoi_scheduling_model

        # genetic algorithm
        self.switches = switches
        self.schedulers_list_smd = schedulers_list_smd
        self.schedulers_list_aoi = schedulers_list_aoi

        # # parameters for animation
        self.animation = animation
        self.freeze_window_after_initsim = freeze_window_after_initsim
        self.freeze_window_at_endsim = freeze_window_at_endsim
        self.tracing = tracing

        # processor lists
        self.smds = []
        self.aois = []

        # state variables
        self.num_jobs = len(self.problem)
        self.jobs_processed = 0
        self.num_forward = 0
        self.total_smd_workload = sum([problem[i]["t_smd"] for i in problem])
        self.unseized_smd_workload = self.total_smd_workload
        self.active_aois = []
        self.passive_aois = []
        self.state = None
        self.observed_job = None
        self.observed_machine = None

        # KPIs
        self.makespan = 0
        self.total_tardiness = 0
        self.num_major_setups = 0

        # embed schedulers
        if smd_scheduling_class and aoi_scheduling_class:
            self.smd_scheduler = smd_scheduling_class(
                hfs_env=self,
                stage="smd",
                noise_model=noise_model
            )
            self.aoi_scheduler = aoi_scheduling_class(
                hfs_env=self,
                stage="aoi",
                noise_model=noise_model
            )
    
            # initialize sequence
            self.sequence = self.smd_scheduler.construct_sequence(self.problem)
        else:
            # embed all required schedulers
            required_schedulers_smd = list(set(self.schedulers_list_smd))
            required_schedulers_aoi = list(set(self.schedulers_list_aoi))
            embedded_schedulers_smd = {}
            embedded_schedulers_aoi = {}

            for scheduler in required_schedulers_smd:
                embedded_schedulers_smd[scheduler] = scheduler(
                    hfs_env=self,
                    stage='smd'
                )

            for scheduler in required_schedulers_aoi:   
                embedded_schedulers_aoi[scheduler] = scheduler(
                    hfs_env=self,
                    stage='aoi'
                )         

            self.embedded_schedulers_smd = embedded_schedulers_smd
            self.embedded_schedulers_aoi = embedded_schedulers_aoi
            self.sequence = self.schedulers_list_smd[0].construct_sequence(self, self.problem)

        # environment creation
        self.define_salabim_env()
        self.create_hfs_env()

        # simulation execution
        self.step_execution = step_execution

        # run simulation if run-by-step mode is not active
        if not self.step_execution:
            self.run_simulation()

    def step(self):

        """
        Run simulation until state is reached. Possible states are:
        - 'smd scheduling', 
        - 'aoi scheduling'
        - 'end of simulation'
        """

        self.state = None
        while self.state is None:
            self.env.step()

        if self.tracing and self.state == "end of simulation":
            print("\n--- end of simulation reached ---\n")

    def define_salabim_env(self):

        self.env = sim.Environment(trace=self.tracing, time_unit="minutes")
        self.env.modelname(
            "Printed Circuit Board Production: A Two-Stage Hybrid Flow Shop Scheduling Problem"
        )
        self.env.animation_parameters(
            animate=self.animation,
            synced=False,
            width=ENV_W,
            height=ENV_H,
            background_color="60%gray",
        )

    def create_hfs_env(self):
        
        # genetic scheduler
        if self.schedulers_list_smd and self.schedulers_list_aoi and self.switches:
            self.geneticscheduler = GeneticScheduler(
                schedulers_list_smd=self.schedulers_list_smd,
                schedulers_list_aoi=self.schedulers_list_aoi,
                embedded_schedulers_smd=self.embedded_schedulers_smd,
                embedded_schedulers_aoi=self.embedded_schedulers_aoi,
                switches=self.switches, hfs_env=self)

        # source
        self.source = Source(name="Source", hfs_env=self)

        # global smd queue
        self.global_smd_queue = Queue(
            name="Global SMD Queue",
            hfs_env=self,
            predecessors=[self.source],
            img_w=GLOBAL_QUEUE_WIDTH,
            img_h=GLOBAL_QUEUE_HEIGHT,
            img_x=X_0 + GLOBAL_SOURCE_DRAIN_RADIUS + 20,
            img_y=Y_GLOBAL_BUFFER,
        )

        # local smd queues
        self.smd_0_queue = Queue(
            name="SMD 01 Queue",
            hfs_env=self,
            predecessors=[self.global_smd_queue],
            img_w=GLOBAL_QUEUE_WIDTH,
            img_h=GLOBAL_QUEUE_HEIGHT / 2,
            img_x=self.global_smd_queue.img_x + self.global_smd_queue.img_w + 20,
            img_y=Y_GLOBAL_BUFFER + 120,
        )

        self.smd_1_queue = Queue(
            name="SMD 02 Queue",
            hfs_env=self,
            predecessors=[self.global_smd_queue],
            img_w=GLOBAL_QUEUE_WIDTH,
            img_h=GLOBAL_QUEUE_HEIGHT / 2,
            img_x=self.global_smd_queue.img_x + self.global_smd_queue.img_w + 20,
            img_y=Y_GLOBAL_BUFFER + 55,
        )

        self.smd_2_queue = Queue(
            name="SMD 03 Queue",
            hfs_env=self,
            predecessors=[self.global_smd_queue],
            img_w=GLOBAL_QUEUE_WIDTH,
            img_h=GLOBAL_QUEUE_HEIGHT / 2,
            img_x=self.global_smd_queue.img_x + self.global_smd_queue.img_w + 20,
            img_y=Y_GLOBAL_BUFFER - 10,
        )

        self.smd_3_queue = Queue(
            name="SMD 04 Queue",
            hfs_env=self,
            predecessors=[self.global_smd_queue],
            img_w=GLOBAL_QUEUE_WIDTH,
            img_h=GLOBAL_QUEUE_HEIGHT / 2,
            img_x=self.global_smd_queue.img_x + self.global_smd_queue.img_w + 20,
            img_y=Y_GLOBAL_BUFFER - 75,
        )

        # smd lines
        self.smd_0 = SMD(
            name="SMD 01",
            hfs_env=self,
            global_queue=self.global_smd_queue,
            local_queue=self.smd_0_queue,
            img_x=self.smd_0_queue.img_x + self.smd_0_queue.img_w + 20,
            img_y=Y_GLOBAL_BUFFER + 90,
            info_x=10,
            info_y=100,
        )

        self.smds.append(self.smd_0)

        self.smd_1 = SMD(
            name="SMD 02",
            hfs_env=self,
            global_queue=self.global_smd_queue,
            local_queue=self.smd_1_queue,
            img_x=self.smd_1_queue.img_x + self.smd_1_queue.img_w + 20,
            img_y=Y_GLOBAL_BUFFER + 25,
            info_x=10,
            info_y=70,
        )

        self.smds.append(self.smd_1)

        self.smd_2 = SMD(
            name="SMD 03",
            hfs_env=self,
            global_queue=self.global_smd_queue,
            local_queue=self.smd_2_queue,
            img_x=self.smd_2_queue.img_x + self.smd_2_queue.img_w + 20,
            img_y=Y_GLOBAL_BUFFER - 40,
            info_x=10,
            info_y=40,
        )

        self.smds.append(self.smd_2)

        self.smd_3 = SMD(
            name="SMD 04",
            hfs_env=self,
            global_queue=self.global_smd_queue,
            local_queue=self.smd_3_queue,
            img_x=self.smd_3_queue.img_x + self.smd_3_queue.img_w + 20,
            img_y=Y_GLOBAL_BUFFER - 105,
            info_x=10,
            info_y=10,
        )

        self.smds.append(self.smd_3)

        # aoi queue
        self.global_aoi_queue = Queue(
            name="Global AOI Queue",
            hfs_env=self,
            predecessors=[self.smd_0, self.smd_1, self.smd_2, self.smd_3],
            img_w=GLOBAL_QUEUE_WIDTH,
            img_h=GLOBAL_QUEUE_HEIGHT,
            img_x=self.smd_0.img_x + GLOBAL_PROCESS_WIDTH + 20,
            img_y=Y_GLOBAL_BUFFER,
        )

        # aoi_lines
        self.aoi_0 = AOI(
            name="AOI 01",
            hfs_env=self,
            global_queue=self.global_aoi_queue,
            img_x=self.global_aoi_queue.img_x + self.global_aoi_queue.img_w + 20,
            img_y=Y_GLOBAL_BUFFER + 130,
        )

        self.aois.append(self.aoi_0)
        self.active_aois.append(self.aoi_0)

        self.aoi_1 = AOI(
            name="AOI 02",
            hfs_env=self,
            global_queue=self.global_aoi_queue,
            img_x=self.global_aoi_queue.img_x + self.global_aoi_queue.img_w + 20,
            img_y=Y_GLOBAL_BUFFER + 65,
        )

        self.aois.append(self.aoi_1)
        self.active_aois.append(self.aoi_1)

        self.aoi_2 = AOI(
            name="AOI 03",
            hfs_env=self,
            global_queue=self.global_aoi_queue,
            img_x=self.global_aoi_queue.img_x + self.global_aoi_queue.img_w + 20,
            img_y=Y_GLOBAL_BUFFER,
        )

        self.aois.append(self.aoi_2)
        self.active_aois.append(self.aoi_2)

        self.aoi_3 = AOI(
            name="AOI 04",
            hfs_env=self,
            global_queue=self.global_aoi_queue,
            img_x=self.global_aoi_queue.img_x + self.global_aoi_queue.img_w + 20,
            img_y=Y_GLOBAL_BUFFER - 65,
        )

        self.aois.append(self.aoi_3)
        self.active_aois.append(self.aoi_3)

        self.aoi_4 = AOI(
            name="AOI 05",
            hfs_env=self,
            global_queue=self.global_aoi_queue,
            img_x=self.global_aoi_queue.img_x + self.global_aoi_queue.img_w + 20,
            img_y=Y_GLOBAL_BUFFER - 130,
        )

        self.aois.append(self.aoi_4)
        self.active_aois.append(self.aoi_4)

        if self.env.animate() == True:
            self.drain = Drain(
                hfs_env=self,
                predecessors=[
                    self.aoi_0,
                    self.aoi_1,
                    self.aoi_2,
                    self.aoi_3,
                    self.aoi_4,
                ],
                img_x=self.aoi_0.img_x + GLOBAL_PROCESS_WIDTH + 75,
                img_y=Y_GLOBAL_SOURCE_DRAIN,
            )
            self.info_makespan = sim.AnimateText(
                text="Makespan: ",
                x=10,
                y=100,
                fontsize=GLOBAL_FONTSIZE,
                screen_coordinates=True,
            )
            self.info_tardiness = sim.AnimateText(
                text="Total Tardiness: 0",
                x=10,
                y=70,
                fontsize=GLOBAL_FONTSIZE,
                screen_coordinates=True,
            )
            self.info_setups = sim.AnimateText(
                text="Major Setups: 0",
                x=10,
                y=40,
                fontsize=GLOBAL_FONTSIZE,
                screen_coordinates=True,
            )
            self.info_violations = sim.AnimateText(
                text="Setup Violations: 0",
                x=10,
                y=10,
                fontsize=GLOBAL_FONTSIZE,
                screen_coordinates=True,
            )
            self.info_num_jobs = sim.AnimateText(
                text="Completed Jobs: 0",
                x=self.drain.img_x - self.drain.img_w,
                y=self.drain.img_y - self.drain.img_w - 20,
                fontsize=GLOBAL_FONTSIZE,
                screen_coordinates=True,
            )

    def run_simulation(self):
        self.env.run()
        
#############################################################################################################################################################################################################

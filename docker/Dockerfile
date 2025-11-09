##############
# Base image #
##############
FROM ros:humble-ros-base AS base

ARG AMENT_WORKSPACE_DIR="/ament_ws"

LABEL org.opencontainers.image.authors="tobit@robots.ox.ac.uk"
LABEL description="ROS 2 Realtime Trees Docker"
LABEL version="1.0"

WORKDIR ${AMENT_WORKSPACE_DIR}
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
   && apt-get -y install \
   git \
   python3-colcon-clean \
   python3-osrf-pycommon \
   python3-rosdep \
   && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=dialog


#####################
# Development image #
#####################
FROM base AS dev

ARG AMENT_WORKSPACE_DIR="/ament_ws"
ARG USERNAME="developer"
ARG UID=1000
ARG GID=1000

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR ${AMENT_WORKSPACE_DIR}

RUN apt-get update \
   && apt-get install -y \
   ack \
   cloc \
   gdb \
   htop \
   iperf3 \
   iputils-ping \
   mlocate \
   nano \
   net-tools \
   psmisc \
   python3-vcstool \
   tmux \
   && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
   && apt-get install -y \
   ros-${ROS_DISTRO}-rqt-image-view \
   ros-${ROS_DISTRO}-rviz2 \
   && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=dialog

RUN apt-get update \
   && apt-get install -y sudo \
   && rm -rf /var/lib/apt/lists/* \
   && addgroup --gid ${GID} ${USERNAME} \
   && adduser --disabled-password --gecos '' --uid ${UID} --gid ${GID} ${USERNAME} \
   && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
   && mkdir -p ${AMENT_WORKSPACE_DIR}/src \
   && chown -R ${UID}:${GID} /home/${USERNAME} \
   && chown -R ${UID}:${GID} ${AMENT_WORKSPACE_DIR}


RUN echo "alias rsource='source ${AMENT_WORKSPACE_DIR}/install/setup.bash'" >> /home/${USERNAME}/.bash_aliases \
   && echo "alias rbuild='(cd ${AMENT_WORKSPACE_DIR} && colcon build)'" >> /home/${USERNAME}/.bash_aliases \
   && echo "alias rclean='(cd ${AMENT_WORKSPACE_DIR} && colcon clean workspace -y)'" >> /home/${USERNAME}/.bash_aliases \
   && echo "rsource || source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USERNAME}/.bashrc

COPY realtime_trees/requirements.txt ${AMENT_WORKSPACE_DIR}/src/realtime-trees/realtime_trees/requirements.txt

RUN apt-get update \
&& apt-get install -y python3-pip \
&& pip3 install --no-cache-dir --timeout 999999 -r ${AMENT_WORKSPACE_DIR}/src/realtime-trees/realtime_trees/requirements.txt \
&& rm -rf /var/lib/apt/lists/*

COPY ../ ${AMENT_WORKSPACE_DIR}/src/realtime-trees/
RUN cd ${AMENT_WORKSPACE_DIR}/src/realtime-trees && pip3 install -e realtime_trees
USER ${USERNAME}

RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build
RUN echo "source ${AMENT_WORKSPACE_DIR}/install/setup.bash" >> /home/${USERNAME}/.bashrc

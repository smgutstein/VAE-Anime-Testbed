#!/usr/bin/env bash
set -euo pipefail

USER_NAME="${USER_NAME:-vaeuser}"
GROUP_NAME="${GROUP_NAME:-vaeuser}"
HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"
USER_HOME="/home/${USER_NAME}"

if ! getent group "${HOST_GID}" >/dev/null 2>&1; then
    groupadd --gid "${HOST_GID}" "${GROUP_NAME}"
else
    GROUP_NAME="$(getent group "${HOST_GID}" | cut -d: -f1)"
fi

if ! id -u "${HOST_UID}" >/dev/null 2>&1; then
    useradd \
        --uid "${HOST_UID}" \
        --gid "${HOST_GID}" \
        --create-home \
        --shell /bin/bash \
        "${USER_NAME}"
else
    USER_NAME="$(getent passwd "${HOST_UID}" | cut -d: -f1)"
    USER_HOME="$(getent passwd "${HOST_UID}" | cut -d: -f6)"
fi

mkdir -p /workspace "${USER_HOME}" "${USER_HOME}/.cache" "${USER_HOME}/.config"
chown -R "${HOST_UID}:${HOST_GID}" "${USER_HOME}"

export HOME="${USER_HOME}"
export USER="${USER_NAME}"
export LOGNAME="${USER_NAME}"

exec gosu "${HOST_UID}:${HOST_GID}" "$@"
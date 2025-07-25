#!/usr/bin/env bash

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2021-2023 Dyne.org foundation <foundation@dyne.org>.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

if [ "${1}" = 'docker' ]
then
    cont_name=nodered_interfacer

    docker rm -f ${cont_name}

    docker run -it -p 1880:1880 -v ${PWD}:/data --name ${cont_name} nodered/node-red
else
    # https://nodered.org/docs/getting-started/local
    ./node_modules/.bin/node-red --settings ./nodered_conf/settings.js
fi
# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup file for AndroidWorld."""

import setuptools
_PROTOBUF_VERSION = '5.29.5'

setuptools.setup(
    name='android_world',
    package_data={
        '': [
            '*.json',
            '*.proto',
            '*.textproto',
            '*.xml',
            'res/xml/*.xml',
        ]
    },
    packages=setuptools.find_packages(),
    install_requires=[
        f'protobuf=={_PROTOBUF_VERSION}',
        'Pillow==10.4.0',
        'pysqlite3-binary==0.5.4',
        'openai',
        'matplotlib==3.6.1',
        'ImageHash',
    ],
)

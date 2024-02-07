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

"""All necessary imports for registration.

Custom models, task, configs, etc need to be imported to registry,so they can be
picked up by the trainer. They can be included in this file so you do not need
to handle each file separately.
"""

# pylint: disable=unused-import
from official.common import registry_imports
import network_config
import network_input
import network_model
import network_task
import network_trainer
from gradient_accumulator import GradientAccumulator

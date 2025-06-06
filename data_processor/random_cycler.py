# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

import random

class RandomCycler:
    """
    Creates an internal copy of a sequence and allows access to its items in a constrained random 
    order. For a source sequence of n items and one or several consecutive queries of a total 
    of m items, the following guarantees hold (one implies the other):
        - Each item will be returned between m // n and ((m - 1) // n) + 1 times.
        - Between two appearances of the same item, there may be at most 2 * (n - 1) other items.
    """
    
    def __init__(self, source):
        if len(source) == 0:
            raise Exception("Can't create RandomCycler from an empty collection")
        self.all_items = list(source)
        self.next_items = []
    
    def sample(self, count: int):
        shuffle = lambda l: random.sample(l, len(l))
        
        out = []
        while count > 0:
            if count >= len(self.all_items):
                out.extend(shuffle(list(self.all_items)))
                count -= len(self.all_items)
                continue
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            count -= n
            self.next_items = self.next_items[n:]
            if len(self.next_items) == 0:
                self.next_items = shuffle(list(self.all_items))
        return out
    
    def __next__(self):
        return self.sample(1)[0]


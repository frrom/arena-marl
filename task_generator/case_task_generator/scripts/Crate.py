#%%
from typing import Iterator
import numpy as np
import warnings
from typing import List, Dict, Union
from geometry_msgs.msg import Pose2D

class Crate:
    def __init__(self, start_location: np.ndarray, end_goal: np.ndarray, index: int):
        self.current_location = start_location
        self.goal = end_goal
        self.index = index
        self.delivered = False

    def __repr__(self):
        return f'Crate index: {self.index}\n\tcurrent location: {self.current_location} | goal: {self.goal} | delivered: {self.delivered}'

    def __call__(self):
        return self.start_location, self.goal, self.index

    def __eq__(self, other):
        if not isinstance(other, Crate):
            return False
        else:
            return other.index == self.index \
                and other.current_location == self.current_location \
                    and other.goal == self.goal

    def move(self, new_location):
        self.current_location = new_location
        self.delivered = True if np.equal(new_location,self.goal).all() else False

    def set_new_goal(self, goal: np.ndarray):
        self.goal = goal
        self.delivered = np.equal(self.current_location, self.goal).all()

    def get_goal(self):
        goal = Pose2D()
        goal.x = self.goal[1]
        goal.y = self.goal[0]
        goal.theta = 1

        return goal


class CrateStack:
    
    
    def __init__(self, name):
        self.name = name
        self.index: int = 0
        self._in_transit: List[Crate] = []
        self._crate_map: Dict[bytes, Crate] = {} # maps current_coords to crate object

    def __repr__(self):
        ret = f'Crate Stack "{self.name}"\n\tActive Crates: {len(self._crate_map)}\n'
        for crate in self._crate_map.values():
            ret += repr(crate) + '\n'

        return ret

        #return f'Crate Stack "{self.name}"\n\tActive Crates: {len(self._crate_map)}'
    
    def __iter__(self) -> Iterator[Crate]:
        return iter(list(self._crate_map.values()))

    def __bool__(self):
        return bool(list(self._crate_map.values()))
    __nonzero__ = __bool__ # python2 backwards compatibility cause we can
    
    def __len__(self):
        return len(list(self._crate_map.values()))

    def __getitem__(self, index) -> Crate:
        return list(self._crate_map.values())[index]

    def _crate_locations_sanity_check(self):
        locs_from_list = [crate.current_location for crate in self._crate_map.values()]
        locs_from_map = [np.frombuffer(key, dtype= int) for key in self._crate_map]
        assert (np.sort(locs_from_list) == np.sort(locs_from_map)).all()

    def get_crate_locations(self, sanity_check=True):
        if sanity_check:
            self._crate_locations_sanity_check()
        return [crate.current_location for crate in self._crate_map.values()]

    def isempty(self):
        return not self.__bool__()

    def add(self, start_location: np.ndarray, end_goal: np.ndarray):
        if start_location.tobytes() in self._crate_map:
            raise ValueError('Can\'t spawn crates on top of eachother')
        crate = Crate(start_location, end_goal, self.index)
        self._crate_map[start_location.tobytes()] = crate
        self.index += 1

        return crate

    def remove(self, location: np.ndarray) -> Crate:
        crate = self._crate_map.pop(location.tobytes())

        return crate

    def move_crate(self, old_location: np.ndarray, new_location: np.ndarray) -> bool:
        if new_location.tobytes() in self._crate_map:
            print(f'Can\'t move crate to {new_location} because there already is one there.')
            return False
        else:
            crate: Crate = self._crate_map.pop(old_location.tobytes()) # np.ndarray.tobytes() makes array hashable
            self._crate_map[new_location.tobytes()] = crate # update map
            crate.move(new_location) # update crate object
            return True

    def get_crate_at_location(self, location: np.ndarray) -> Crate:
        crate = self._crate_map.get(location.tobytes(), None)
        if crate is None:
            print(f'No crate found at location {location}')
        return crate

    def pickup_crate(self, crate_location: np.ndarray) -> Crate:
        """
        If there's crate at location, remove it from the _crate_map and add it to the _in_transit list.
        
        params:
        --------
        crate_location - Location to pick up crate from

        returns:
        --------
        Index of picked up crate, if crate get's picked up.
        None, if crate wasn't picked up. (Probably no Crate at crate_location)

        exceptions:
        -----------
        If there's no crate at location: WARNING CrateWarning and return None.
        """
        try:
            crate = self.remove(crate_location)
            self._in_transit.append(crate)
            return crate

        except KeyError as e:
            warnings.warn(f'{crate_location} does not have a Crate.', CrateWarning)

    def drop_crate(self, crate_index: int, drop_location: np.ndarray):
        """
        If the crate with crate_index is in transit, drop it at drop_location.

        If drop_location is occupied or crate with crate_index is not in transit, warn CrateWarning.
        """
        for crate in self._in_transit:
            if crate.index == crate_index:
                if drop_location.tobytes() in self._crate_map:
                    warnings.warn("Can't drop crate on top of other crate", CrateWarning)
                else:
                    self._crate_map[drop_location.tobytes()] = crate
                    self._in_transit.remove(crate)
                    crate.move(drop_location)
                return True
        else:
            warnings.warn("Crate is not in transit.", CrateWarning)
            return False



class CrateWarning(UserWarning):
    pass




# %%
if __name__ == '__main__':
    cs = CrateStack('One')
    crate_1 = cs.spawn_crate(np.array([0,0]), np.array([1,1]))
    cs.list
    # %%
    cs.despawn_crate(crate_1)
    cs.list

# %%

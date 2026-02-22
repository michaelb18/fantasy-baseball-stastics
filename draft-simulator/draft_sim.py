class Player:
    def __init__(self, name: str, salary: float, position: str):
        self.name = name
        self.salary = salary
        self.position = position

class Batter(Player):
    def __init__(self, name: str, salary: float, position: str, team_name: str, hr: int, rbi: int, r: float, obp: float, sb: int):
        super().__init__(name, salary, position)

        self.hr = hr
        self.rbi = rbi
        self.r = r
        self.obp = obp
        self.sb = sb

class Pitcher(Player):
    def __init__(self, name: str, salary: float, position: str, team_name: str, k: int, era: int, wins: int, whip: float, svhld: float):
        super().__init__(name, salary, position)
        
        self.k = k
        self.era = era
        self.wins = wins
        self.whip = whip
        self.svhld = svhld

class Movie:
       def __init__(self, ID:int, name:str, year:int, genres, director = ''):
              self.title = name
              self.ID = ID
              self.year = year
              self.genres = genres
              self.director = director

       def __str__(self):
              return f"{self.title}"
       
       def info(self):
              return [self.ID, self.title, self.director ,self.year, self.genres]
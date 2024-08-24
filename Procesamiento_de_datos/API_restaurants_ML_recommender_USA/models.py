from sqlalchemy import create_engine, Column, Integer, String, Float, INTEGER, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship



#engine = create_engine('mysql+pymysql://root:root@localhost/movies')
engine = create_engine('sqlite:///data/restaurants.db')

conn = engine.connect()
Base = declarative_base()


# Crear las tablas en la base de datos
Base.metadata.create_all(engine)

# Crear la sesión
Session = sessionmaker(bind=engine)
session = Session()

Session = sessionmaker(engine)
session = Session()



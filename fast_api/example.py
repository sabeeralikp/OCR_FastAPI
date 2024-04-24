# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy import Column, Integer, String
# from sqlalchemy.ext.declarative import declarative_base


# Base = declarative_base()


# class User(Base):
#     __tablename__ = "users"
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     name = Column(String)


# engine = create_engine("sqlite:///example.db")
# Session = sessionmaker(bind=engine)
# session = Session()

# user = User(name="John Doe")
# session.add(user)
# session.commit()

# print(user.id)

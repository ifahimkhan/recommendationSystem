import pymysql

def getConnect():
    return pymysql.connect('localhost','root',"",'test')

def insertUser(t):
    db=getConnect()
    cr=db.cursor()
    sql="insert into user1 values(%s,%s,%s,%s,%s)"
    cr.execute(sql,t)
    db.commit()
    db.close()

def updateUser(t):
    db=getConnect()
    cr=db.cursor()
    sql="update user1 set name=%s,contact=%s,address=%s,passw=%s where id=%s"
    cr.execute(sql,t)
    db.commit()
    db.close()



def userDelete(id):
    db=getConnect()
    cr=db.cursor()
    sql="delete from user1 where id=%s"
    cr.execute(sql,id)
    db.commit()
    db.close()


def getAllUser():
    db=getConnect()
    cr=db.cursor()
    sql="select * from user1"
    cr.execute(sql)
    ls=cr.fetchall()
    db.commit()
    db.close()
    return ls

def getUserById(id):
    db=getConnect()
    cr=db.cursor()
    sql="select * from user1 where id=%s"
    cr.execute(sql,id)
    ls=cr.fetchall()
    db.commit()
    db.close()
    return ls[0]




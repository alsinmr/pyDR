from multiprocessing.connection import Listener,Client
listener=Listener(("localhost",6001),authkey=b"pyDIFRATE password")
client=Client(("localhost",7001),authkey=b"pyDIFRATE password")
conn=listener.accept()
while True:
	fun=conn.recv()
	if hasattr(fun,"__call__"):
		conn.close()
		break
client.send("received")
client.close()
fun()

import mysql.connector
import openpyxl

def write_to_db():
	wb = openpyxl.load_workbook('handrail.xlsx')
	ws=wb.active

	mydb = mysql.connector.connect(
		host='localhost',
		user='adminpoleng',
		password='adminpoleng',
		database='handrail'
	)

	mycursor=mydb.cursor()

	sql = ('INSERT INTO handrail_violation (class ,time) VALUES (%s, %s)')
	data = []
	for row in ws.iter_rows(values_only=True):
		data.append(list(row))

	print(data)

	mycursor.executemany(sql, data)

	mydb.commit()

	print(mycursor.rowcount, 'was inserted.')
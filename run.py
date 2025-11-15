from app import create_app

print("Creating app...")
app=create_app()
print("App Created....")

if __name__ == "__main__":
     app.run(host="0.0.0.0",port=5000,debug=True) 
import os
import smtplib, ssl

from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd

from menu_solver import MenuSolver

load_dotenv()

sender_email = os.getenv("SENDER_EMAIL")
receiver_email = os.getenv("RECEIVER_EMAIL")
password = os.getenv("PASSWORD")

base_url = os.getenv("SPREADSHEET_URL")
difficulty_df = pd.read_csv(
    base_url.format(os.getenv("DIFFICULTY_ID")),
    skiprows=1,
    names=["menu_id", "difficulty"],
)
ingredient_df = pd.read_csv(
    base_url.format(os.getenv("INGREDIENT_ID")),
    skiprows=1,
    names=["menu_id", "ingredient"],
)
category_df = pd.read_csv(
    base_url.format(os.getenv("CATEGORY_ID")),
    skiprows=1,
    names=["menu_id", "category"],
)
menu_df = pd.read_csv(
    base_url.format(os.getenv("MENU_ID")), skiprows=1, names=["menu_id", "waktu"]
)
difficulty_dict = np.vectorize(
    difficulty_df.set_index("menu_id").difficulty.to_dict().get
)

n_days = 14
n_candidates = 10
n_fittest = 5
n_iter = 100
p_mutation = 0.1

solver = MenuSolver(
    menu_df, difficulty_dict, n_days, n_candidates, n_fittest, n_iter, p_mutation
)
output_table = solver.run()

message = MIMEMultipart("alternative")
message["Subject"] = "multipart test"
message["From"] = sender_email
message["To"] = receiver_email

text = f"""\
Hi,
Ini adalah daftar menu {n_days/7:.0f} minggu ke depan.
{output_table.to_markdown()}
"""
html = f"""\
<html>
  <body>
    <p>Hi,<br>
       Ini adalah daftar menu {n_days/7:.0f} minggu ke depan.<br>
    </p>
    {output_table.to_html()}
  </body>
</html>
"""

# Turn these into plain/html MIMEText objects
part1 = MIMEText(text, "plain")
part2 = MIMEText(html, "html")

# Add HTML/plain-text parts to MIMEMultipart message
# The email client will try to render the last part first
message.attach(part1)
message.attach(part2)

# Create secure connection with server and send email
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, message.as_string())

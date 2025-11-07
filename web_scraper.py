import requests
from bs4 import BeautifulSoup
import csv
import webbrowser

# -----------------------------
# Step 1: Target website
# -----------------------------
url = "https://www.nba.com/teams"
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

# -----------------------------
# Step 2: Find all team sections
# -----------------------------
teams = soup.find_all('div', class_='TeamFigure_tf__jA5HW')

# -----------------------------
# Step 3: Create CSV file
# -----------------------------
with open('nba_teams.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Team Name', 'Logo URL', 'Profile Link', 'Stats Link', 'Schedule Link', 'Tickets Link'])

    # -----------------------------
    # Step 4: Extract data for each team
    # -----------------------------
    for team in teams:
        # Team name
        name_tag = team.find('a', class_='TeamFigure_tfMainLink__OPLFu')
        team_name = name_tag.text.strip() if name_tag else 'N/A'

        # Logo URL
        logo_tag = team.find('img')
        logo_url = logo_tag['src'] if logo_tag else 'N/A'

        # Sub-links (Profile, Stats, Schedule, Tickets)
        links_div = team.find('div', class_='TeamFigure_tfLinks__gwWFj')
        links = links_div.find_all('a') if links_div else []

        profile, stats, schedule, tickets = ['N/A'] * 4
        if len(links) >= 4:
            profile = "https://www.nba.com" + links[0]['href']
            stats = "https://www.nba.com" + links[1]['href']
            schedule = "https://www.nba.com" + links[2]['href']
            tickets = "https://www.nba.com" + links[3]['href']

        # Write one row per team
        writer.writerow([team_name, logo_url, profile, stats, schedule, tickets])

print(" Data scraped successfully and saved in 'nba_teams.csv'")

"""
generate_passwords.py — One-time administrator utility for RGU ONA tool.

Run this script once before deploying the app to replace the placeholder "TBD"
passwords in config.yaml with securely hashed bcrypt passwords.

Usage:
    python generate_passwords.py

The script will prompt for a new password for each user defined in config.yaml,
hash all passwords using streamlit-authenticator's Hasher, and write the hashed
values back into config.yaml.  The original plain-text passwords are never stored.

IMPORTANT:
    - Run this from the project root directory (same folder as config.yaml).
    - Never commit config.yaml to a public repository after running this script.
    - If you need to reset a password, edit config.yaml to set password: TBD for
      that user and run this script again.
"""

import sys
import os

import yaml
from yaml.loader import SafeLoader

CONFIG_PATH = "config.yaml"


def main():
    # ------------------------------------------------------------------ #
    # Load config.yaml                                                     #
    # ------------------------------------------------------------------ #
    if not os.path.exists(CONFIG_PATH):
        print(f"ERROR: {CONFIG_PATH} not found. "
              "Please create it from the project template first.")
        sys.exit(1)

    with open(CONFIG_PATH, "r") as f:
        config = yaml.load(f, Loader=SafeLoader)

    usernames = config.get("credentials", {}).get("usernames", {})
    if not usernames:
        print("ERROR: No usernames found in config.yaml.")
        sys.exit(1)

    print("=" * 60)
    print("RGU ONA Tool — Password Setup")
    print("=" * 60)
    print("You will be prompted to enter a password for each user.")
    print("Passwords will be hashed with bcrypt before being saved.")
    print()

    # ------------------------------------------------------------------ #
    # Collect new plain-text passwords from the administrator             #
    # ------------------------------------------------------------------ #
    for username, user_data in usernames.items():
        display_name = user_data.get("name", username)
        email = user_data.get("email", "")
        print(f"User: {display_name} ({username}) <{email}>")
        while True:
            password = input(f"  Enter password for '{username}': ").strip()
            if password:
                break
            print("  Password cannot be empty. Please try again.")
        # Store temporarily as plain text; Hasher will hash it next
        usernames[username]["password"] = password
        print()

    # ------------------------------------------------------------------ #
    # Hash all passwords using streamlit-authenticator Hasher             #
    # API confirmed for version 0.4.x:                                    #
    #   Hasher.hash_passwords(credentials_dict) -> credentials_dict       #
    #   Takes the full credentials dict (with 'usernames' key).           #
    # ------------------------------------------------------------------ #
    try:
        import streamlit_authenticator as stauth

        # Hasher.hash_passwords is a classmethod in 0.4.x
        # It accepts config['credentials'] (the dict with 'usernames' key)
        config["credentials"] = stauth.Hasher.hash_passwords(config["credentials"])
        print("Passwords hashed successfully using streamlit-authenticator Hasher.")

    except AttributeError:
        # Fallback for older API variants that use Hasher([passwords]).generate()
        try:
            import streamlit_authenticator as stauth
            plain_passwords = [
                usernames[u]["password"] for u in usernames
            ]
            hashed = stauth.Hasher(plain_passwords).generate()
            for username, hashed_pw in zip(usernames.keys(), hashed):
                config["credentials"]["usernames"][username]["password"] = hashed_pw
            print("Passwords hashed successfully using Hasher.generate() (legacy API).")
        except Exception as exc:
            print(f"ERROR: Could not hash passwords: {exc}")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Write hashed config back to config.yaml                             #
    # ------------------------------------------------------------------ #
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print()
    print("Passwords hashed and saved to config.yaml successfully.")
    print()
    print("NEXT STEPS:")
    print("  1. Verify config.yaml contains bcrypt hashes (they start with $2b$).")
    print("  2. Change the cookie key in config.yaml to a long random string.")
    print("  3. Do NOT commit config.yaml to a public Git repository.")
    print("  4. Start the app:  streamlit run app.py")


if __name__ == "__main__":
    main()

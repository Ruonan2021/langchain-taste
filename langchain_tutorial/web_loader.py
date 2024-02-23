import hashlib
import sys
from datetime import datetime

import requests
from bs4 import BeautifulSoup


def get_soup(link):
    """
    Return the BeautifulSoup object for input link
    """
    request_object = requests.get(link, auth=("user", "pass"))
    soup = BeautifulSoup(request_object.content, "lxml")
    return soup


def get_status_code(link):
    """
    Return the error code for any url
    param: link
    """
    try:
        error_code = requests.get(link).status_code
    except requests.exceptions.ConnectionError:
        error_code = -1
    return error_code


def find_internal_urls(main_url, depth=0, max_depth=2):
    all_urls_info = []

    soup = get_soup(main_url)
    a_tags = soup.findAll(
        "a",
        {
            "class": [
                "sidebar-item-text sidebar-link active",
                "sidebar-item-text sidebar-link",
            ]
        },
        href=True,
    )

    if main_url.endswith("/"):
        domain = main_url
    else:
        domain = "/".join(main_url.split("/")[:-1])

    if depth > max_depth:
        return {}
    else:
        for a_tag in a_tags:
            if (
                "http://" not in a_tag["href"]
                and "https://" not in a_tag["href"]
                and "/" in a_tag["href"]
            ):

                url = domain.rstrip("docs/") + a_tag["href"].lstrip("..")
            elif "http://" in a_tag["href"] or "https://" in a_tag["href"]:
                url = a_tag["href"]
            else:
                continue

            # status_dict = {}
            # status_dict["url"] = url
            # status_dict["status_code"] = get_status_code(url)
            # status_dict["timestamp"] = datetime.now()
            # status_dict["depth"] = depth + 1
            # all_urls_info.append(status_dict)
            all_urls_info.append(url)

    return all_urls_info


url = "https://shiny.posit.co/py/docs/quick-start.html"

url_list = find_internal_urls(url)

url_list

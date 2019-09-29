#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Thesis: S(peech)T(hought)W(riting)R(epresentation) recognition
author: Luise Schricker

python-3.4

This file contains methods for visualizing a text with annotations of STWR types and speech, thought, writing classes.
"""

def visualize_html(text, labels, filename):
    """
    Method for visualizing a text annotated with STWR types and speech, thought and writing as HTML file.

    :param text: The original text.
    :param labels: A list of labels (format:"direct_speech,2,10").
    :param filename: The filename without extension for th HTML file.
    """

    with open('{}.html'.format(filename), 'w', encoding='utf-8') as f:

        # Definitions of colors and styles to denote annotations
        colors = {
            "direct": "#3498DB",
            "indirect": "#58D68D",
            "free_indirect": "#EC7063",
            "reported": "#F1C40F",
        }

        styles = {
            "speech": ("<u>", "</u>"),
            "thought": ("<b>", "</b>"),
            "writing": ("<i>", "</i>")
        }

        # Iterate over label and modify text with html tags that show the annotations.
        # The number of characters that have been added via tags have to be kept in order to keep the character based spans up-to-date
        added_chars = 0

        # Sort labels, so that labels for the same segment are together
        labels_sorted = [[l for l in labels if l.split(",")[1] == l_hat.split(",")[1]] for l_hat in labels]
        # Delete duplicates
        labels_set = set(map(tuple, labels_sorted))  # need to convert the inner lists to tuples so they are hashable
        labels_sorted = list(map(list, labels_set))
        labels_sorted.sort(key = lambda x: int(x[0].split(",")[1]))

        for idx, labels_idx in enumerate(labels_sorted):

            for idx2, label in enumerate(labels_idx):
                label_split = label.split(",")

                rep_type = label_split[0].split("_")[0]
                stw_type = label_split[0].split("_")[1]

                start = int(label_split[1]) + added_chars
                end = int(label_split[2]) + added_chars

                if idx2 == 0:
                    annotation = "<font color={}>{}{}{}</font>".format(colors[rep_type], styles[stw_type][0],text[start:end] , styles[stw_type][1])
                else:
                    annotation = "<font color={}>{}{}{}</font>".format(colors[rep_type], styles[stw_type][0], annotation , styles[stw_type][1])

            added_chars += len(annotation) - len(text[start:end])
            text = text[0:start] + annotation + text[end:]
            annotation = ""

        text = text.replace("\n", "</br>")

        html = """
        <!DOCTYPE html>
        <html lang="de">
        <html>
        <head><meta charset="utf-8"/></head>
        <body><p style="font-family:verdana;">
        KEY:</br>
        <p style="color:{};">Direct STWR</p>
        <p style="color:{};">Indirect STWR</p>
        <p style="color:{};">Free indirect STWR</p>
        <p style="color:{};">Reported STWR</p>
        <p><u>Speech</u></p>
        <p><b>Thought</b></p>
        <p><i>Writing</i></p>
        <hr>
        </br></br>
        {}
        </p></body>
        </html>
        """.format(colors["direct"], colors["indirect"], colors["free_indirect"], colors["reported"], text)

        f.write(html)

# Execution
if __name__ == "__main__":
    visualize_html("Lebe wohl, sagte sie mit Thränen, du Schauplatz meines Glücks, ihr rauschenden Bäume, die ihr mich und den Geliebten oft in eurem dunkeln Schatten vor den spähenden Augen verbarget! Bla blab blabababab", ["direct_speech,0,9", "direct_speech,34,73", "direct_speech,73,191"], "test")
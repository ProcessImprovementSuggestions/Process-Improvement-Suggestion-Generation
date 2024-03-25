# Tell me What to Do: Automatically Generating Process Improvement Suggestions
This repository contains the prototype of the technique described in Tell me What to Do: Automatically Generating Process Improvement Suggestions. Under submission for the International Conference on Business Process Management 2024. 

## Description
Process mining techniques play an important role for understanding, analyzing, and improving business processes. Despite their value, deriving actionable improvement measures from process mining insights remains challenging, requiring manual analysis by process analysts. Existing approaches and frameworks provide abstract suggestions, necessitating translation into actionable solutions. Recent efforts focus on generating alternative execution paths rather than concrete improvement suggestions, leaving process improvement a labor-intensive task. Addressing this gap, we propose a natural language-driven technique leveraging Large Language Models (LLMs) and social media posts as a rich information source for business-to-consumer (B2C) processes. Our technique identifies process weaknesses from social media posts and generates improvement suggestions using multiple knowledge resources. An evaluation against manually annotated posts demonstrates the effectiveness of our approach, producing suggestions perceived as more useful than human-generated ones. Each suggestion is traceable to its source, enhancing explainability and validity. Furthermore, our technique allows to adapt its knowledge base, allowing seamless integration of additional knowledge sources. Thus, it offers a promising avenue to automate and streamline process redesign efforts across diverse contexts, reducing manual effort in the business process management lifecycle.

### Built with
* ![platform](https://img.shields.io/badge/platform-linux-brightgreen)
* ![GPU](https://img.shields.io/badge/GPU-Nvidia%20A10-red)
* ![python](https://img.shields.io/badge/python-black?logo=python&label=3.11.5)

You need a running *qdrant* service (see [documentation](https://qdrant.tech)) and a running *grobid* service (see [documentation](http://grobid.readthedocs.io/)) .

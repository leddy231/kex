class LatexTable:
    
    def __init__(self):
        self.headers = []
        self.emptyFirstHeader = False
        self.rows = []
        self.nrColumns = None
        self.boldHeaders = True
        self.boldIndexColumn = True
        self.columnAlignments = None
        self.customHeader = None

    def render(self):
        res = ""

        if self.customHeader is None:
            headers = self.headers.copy()
            if self.boldHeaders:
                headers = list(map(lambda h: "\\textbf{"+h+"}", headers))
            if self.emptyFirstHeader:
                headers = ["\multicolumn{1}{c|}{}"] + headers

        if self.nrColumns is None:
            self.nrColumns = len(headers)

        if self.columnAlignments is None:
            self.columnAlignments = ['l'] * self.nrColumns
        
        if type(self.columnAlignments) == str:
            self.columnAlignments = ['l'] + [self.columnAlignments] * (self.nrColumns - 1)
        

        alignments = '|'.join(self.columnAlignments)
        res += "\\begin{tabular}{|"+alignments+"|} "

        if self.emptyFirstHeader:
            res += "\cline{2-"+str(self.nrColumns)+"}\n"
        else:
            res += "\hline\n"

        if self.customHeader is None:
            res += " & ".join(headers) + " \\\\ \\hline\n"
        else:
            res += self.customHeader

        for i, row in enumerate(self.rows):
            if row == '!boldLine':
                continue

            
            if i+1 < len(self.rows):
                nxt = self.rows[i+1]
            else:
                nxt = None

            if row == '!emptyRow':
                row = '\multicolumn{'+str(self.nrColumns)+'}{c}{}'
            else:
                row = row.copy()
                if self.boldIndexColumn:
                    row[0] = "\\textbf{"+row[0]+"}"
                row = ' & '.join(row)
            if nxt == '!boldLine':
                row += ' \\\\ \Xhline{3\\arrayrulewidth}\n'
            elif row == '!emptyRow' and nxt is None:
                row += " \\\\ \n"
            else:
                row += " \\\\ \hline\n"
                

            res += row
        res += "\end{tabular}"
        return res
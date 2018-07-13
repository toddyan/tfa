from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from urllib.parse import urlsplit,parse_qs,unquote_plus

host = ('localhost', 8888)
class Resquest(BaseHTTPRequestHandler):
    def do_GET(self):
        resp = {'code': '0','msg':'ok'}
        path = self.path
        query = parse_qs(urlsplit(path).query)
        url = ""
        vid = ""
        for k,v in query.items():
            if k == 'url' and len(v)==1:
                url = unquote_plus(v[0])
            if k == 'vid' and len(v)==1:
                vid = unquote_plus(v[0])
        print("query is:", query, url, vid)
        if url == '' or vid == '':
            resp = {'code': '1', 'msg': 'missing url or vid'}
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

if __name__ == '__main__':
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()
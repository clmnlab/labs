function response = slack_incoming(url, varargin)
%  from https://gist.github.com/gkhtnk/71e0570705944ff04adb6729822798a1
%% response = slack_incoming(url, payload)
% url      : [char]  Webhook URL (you have to get Webhook URL from Slack)
% payload  : [char, struct]  payload must be JSON style string or struct
%            # payload is a case when varagin is a single argument 
% varargin : [char/char] optional mode for define payload as key/value pair
% 
% response : 'ok'        POST of payload was succeeded.
%            otherwise   failed to POST
% 
% see also Slack Incoming Webhooks <https://api.slack.com/incoming-webhooks>





% The MIT License (MIT)
% 
% Copyright (c) 2017 Gaku Hatanaka
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy 
% of this software and associated documentation files (the "Software"), to deal 
% in the Software without restriction, including without limitation the rights 
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
% copies of the Software, and to permit persons to whom the Software is 
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
% SOFTWARE.


% 1. Get Webhook URL from https://slack.com/services/new/incoming-webhook.
% 2. you can get URL like https://hooks.slack.com/services/TXXXXX/BXXXXX/XXXXXXXXXX
% 3. Download slack_incoming.matlab from my gist.
% Execute as follows.
% 
% Example
% >> url = 'https://hooks.slack.com/services/TXXXXX/BXXXXX/XXXXXXXXXX';
% >> slack_incoming(url, 'text', 'text from matlab', 'channel', '#matlab');
%
if nargin == 2
  payload = varargin{1};
  
elseif nargin > 2
  if mod(nargin-1, 2)
    error('slack_incoming:KEY_VALUE_PAIR', 'varargin must be ''key'', ''value''} pair');
  end
  
  payload = struct();
  for fi = 1:(nargin-1)/2
    if ~ischar(varargin{fi*2-1}) || ~ischar(varargin{fi*2})
      error('slack_incoming:ALL_KEY_VALUE_MUST_BE_CHAR', 'All key/value must be char');
    end
    payload.(varargin{fi*2-1}) = varargin{fi*2};
  end
end


if isstruct(payload)
  % case : payload defined as a matlab struct
  % Define "payload" struct according to the JSON format specified by the Slack Incoming Webhooks API
  % "payload" struct must have at least "text" field
  field = fieldnames(payload);
  NofField = length(field);
  
  if ~any(cellfun(@(f) any(strcmp(f, {'text', 'attachments'})), field))
    error('slack_incoming:INCLUDE_TEXT_FIELD_IN_PAYLOAD', '"text" field must be include in "payload" struct');
  end
  
  str = '{';
  for fi = 1:NofField
    if ~ischar(payload.(field{fi}))
      error('slack_incoming:ALL_FIELD_MUST_BE_CHAR', 'All field must be char');
    end
    
    if strcmp(field{fi}, 'attachments')
      pair = strcat('"', field{fi}, '":', payload.(field{fi}));
    else
      pair = strcat('"', field{fi}, '":"', payload.(field{fi}), '"');
    end
    
    if fi ~= NofField
      str = strcat(str, pair, ',');
    else
      str = strcat(str, pair, '}');
    end
  end
  
elseif ischar(payload)
  % case : payload defined as str(JSON format)
  % This code doesn't check "payload" string
  str = payload;
  
else
  error('slack_incoming:CHECK_PAYLOAD', '"payload" must be str(JSON) or struct');
end


response = webwrite(url, str);
if ~strcmp(response, 'ok')
  error('slack_incoming:FAILD_TO_POST', 'failed to post');
end
#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import date, datetime, timedelta
from itertools import cycle
import os
import requests
import shutil
import socket
import tempfile

import pandas as pd
import numpy as np

import imageio
from tqdm import tqdm

PREFIX = 'C:\\Users\\watso' if socket.gethostname() == 'DESKTOP-VD3TK5G' else 'K:\\'


# In[2]:


from bokeh.plotting import figure, curdoc

from bokeh.io import export_png
from bokeh.models import ColumnDataSource, CustomJS, Panel, Tabs, ColorBar, LinearColorMapper,     LogColorMapper, LogTicker, BasicTicker
from bokeh.models.axes import LinearAxis, LogAxis
from bokeh.models.widgets import CheckboxGroup, Dropdown, RadioGroup, MultiSelect, DatePicker, Button, Tabs, Panel
from bokeh.events import MenuItemClick

from bokeh.layouts import column, row
from bokeh.palettes import viridis, Category20_20, linear_palette, Turbo256, Viridis256, Inferno256, Plasma256

from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application

from bokeh.sampledata.us_states import data as US_STATES
from bokeh.sampledata.us_counties import data as US_COUNTIES

if 'HI' in US_STATES: del US_STATES['HI']
if 'AK' in US_STATES: del US_STATES['AK']

PALETTE = Plasma256
    
# output_notebook()


# In[3]:


def compute_states_data():
    GH_STATES_DATA.sort_values('date')
    for fips in tqdm(GH_STATES_DATA['fips'].unique()):

        slicer = GH_STATES_DATA['fips'] == fips
        subset = GH_STATES_DATA.loc[slicer, :]

        state = subset['state'].values[0]
        pop = population(state)

        avg_dates = subset['date'] - (timedelta(days=3) + timedelta(hours=12))
        diff_cases = subset['cases'].diff()
        diff_deaths = subset['deaths'].diff()
        avg_cases = subset['cases'].diff().rolling(7).mean()
        avg_deaths = subset['deaths'].diff().rolling(7).mean()

        GH_STATES_DATA.loc[subset.index, 'diff_cases'] = diff_cases
        GH_STATES_DATA.loc[subset.index, 'diff_deaths'] = diff_deaths
        GH_STATES_DATA.loc[subset.index, 'diff_cases_pc'] = diff_cases / pop * 100000
        GH_STATES_DATA.loc[subset.index, 'diff_deaths_pc'] = diff_deaths / pop * 100000
        GH_STATES_DATA.loc[subset.index, 'avg_dates'] = avg_dates
        GH_STATES_DATA.loc[subset.index, 'avg_cases'] = avg_cases
        GH_STATES_DATA.loc[subset.index, 'avg_deaths'] = avg_deaths
        GH_STATES_DATA.loc[subset.index, 'avg_cases_pc'] = avg_cases / pop * 100000
        GH_STATES_DATA.loc[subset.index, 'avg_deaths_pc'] = avg_deaths / pop * 100000


# In[4]:


def compute_counties_data():
    
    subset = GH_COUNTIES_DATA.loc[:, ('state', 'county')]
    state_county = set()
    for index, row in tqdm(subset.iterrows()):
        if row['county'].lower() != 'unknown':
            state_county.add((row['state'], row['county']))
    state_county = sorted(state_county)    

    GH_COUNTIES_DATA.sort_values('date')
    for state, county in tqdm(state_county):

        slicer = (GH_COUNTIES_DATA['state'] == state).values & (GH_COUNTIES_DATA['county'] == county).values
        subset = GH_COUNTIES_DATA.loc[slicer, :]

        pop = population(f'{state}, {county}')

        avg_dates = subset['date'] - (timedelta(days=3) + timedelta(hours=12))
        diff_cases = subset['cases'].diff()
        diff_deaths = subset['deaths'].diff()
        avg_cases = subset['cases'].diff().rolling(7).mean()
        avg_deaths = subset['deaths'].diff().rolling(7).mean()

        GH_COUNTIES_DATA.loc[subset.index, 'diff_cases'] = diff_cases
        GH_COUNTIES_DATA.loc[subset.index, 'diff_deaths'] = diff_deaths
        GH_COUNTIES_DATA.loc[subset.index, 'diff_cases_pc'] = diff_cases / pop * 100000
        GH_COUNTIES_DATA.loc[subset.index, 'diff_deaths_pc'] = diff_deaths / pop * 100000
        GH_COUNTIES_DATA.loc[subset.index, 'avg_dates'] = avg_dates
        GH_COUNTIES_DATA.loc[subset.index, 'avg_cases'] = avg_cases
        GH_COUNTIES_DATA.loc[subset.index, 'avg_deaths'] = avg_deaths
        GH_COUNTIES_DATA.loc[subset.index, 'avg_cases_pc'] = avg_cases / pop * 100000
        GH_COUNTIES_DATA.loc[subset.index, 'avg_deaths_pc'] = avg_deaths / pop * 100000


# In[5]:


POP_DATA = pd.read_csv('pop_data.csv')


# In[34]:


EMPTY_COUNTIES = {'Alaska': ['Borough', 'Census Area'],
                  'District of Columbia': ['District of Columbia'],
                  'Maryland': ['Baltimore city'],
                  'Virginia': ['Virginia Beach city', 'Alexandria city', 'Harrisonburg city', 'Charlottesville city',
                               'Williamsburg city', 'Richmond city', 'Newport News city', 'Norfolk city',
                               'Portsmouth city', 'Suffolk city', 'Danville city', 'Chesapeake city',
                               'Fredericksburg city', 'Manassas city', 'Hampton city', 'Lynchburg city',
                               'Poquoson city', 'Radford city', 'Bristol city', 'Galax city',
                               'Roanoke city', 'Hopewell city', 'Manassas Park city', 'Winchester city',
                               'Petersburg city', 'Franklin city', 'Waynesboro city', 'Salem city',
                               'Buena Vista city', 'Emporia city', 'Lexington city', 'Staunton city',
                               'Colonial Heights city', 'Fairfax city', 'Falls Church city',
                               'Norton city', 'Covington city'],
                  'Nevada': ['Carson City'],
                  'Missouri': ['St. Louis city'],}
REPLACE_COUNTIES = {'Alaska': {'Anchorage': 'Anchorage Municipality, Alaska'},
                    'New York': {'New York City': 'New York County, New York'},
                    'New Mexico': {'Doña Ana': 'Do�a Ana County, New Mexico'}}

def format_region_name(region):
    if ', ' in region:
        state, county = region.split(', ')
        county_name = 'County' if state != 'Louisiana' else 'Parish'
        if state in EMPTY_COUNTIES and (county in EMPTY_COUNTIES[state] or
                                        any(val in county for val in EMPTY_COUNTIES[state])):
            region = f'{county}, {state}'
        elif state in REPLACE_COUNTIES and county in REPLACE_COUNTIES[state]:
            region = REPLACE_COUNTIES[state][county]
        else:
            region = f'{county} {county_name}, {state}'
    return region

def parse_detailed_name(name):
    part = ' County, ' if ' Parish, Louisiana' not in name else ' Parish, '
    county, _, state = name.partition(part)
    if state == 'New York' and county in ['Queens', 'Kings', 'New York', 'Richmond', 'Bronx']:
        county = 'New York City'
    return state, county

def get_pop_entry(region):
    region = format_region_name(region)
    entry = POP_DATA[POP_DATA['NAME'] == region]
    return entry
    
def population(region):
    if region == 'Missouri, Joplin':
        return 50657
    if region == 'Missouri, Kansas City':
        return 491918
    entry = get_pop_entry(region)
    try:
        return int(entry.values[0][2])
    except:
        raise Exception(f'Unable to find population of {region}!')


# In[7]:


gh_states_data_file = os.path.join(PREFIX, 'covid-19-data', 'us-states.csv')
gh_counties_data_file = os.path.join(PREFIX, 'covid-19-data', 'us-counties.csv')

drop_states = ['Guam', 'Northern Mariana Islands', 'Virgin Islands', 'Puerto Rico']
drop_counties = drop_states + ['Hawaii', 'Alaska']

if not os.path.exists('us-states.csv') or (os.path.exists(gh_states_data_file) and 
     os.stat(gh_states_data_file).st_mtime > os.stat('us-states.csv').st_mtime):
    GH_STATES_DATA = pd.read_csv(gh_states_data_file, parse_dates=['date'])
    for state in drop_states:
        GH_STATES_DATA.drop(GH_STATES_DATA[GH_STATES_DATA['state'] == state].index, inplace=True)
    compute_states_data()
    GH_STATES_DATA.to_csv('us-states.csv')
else:
    GH_STATES_DATA = pd.read_csv('us-states.csv', parse_dates=['date', 'avg_dates'])
    
if not os.path.exists('us-counties.csv') or (os.path.exists(gh_counties_data_file) and 
     os.stat(gh_counties_data_file).st_mtime > os.stat('us-counties.csv').st_mtime):
    GH_COUNTIES_DATA = pd.read_csv(gh_counties_data_file, parse_dates=['date'])
    for state in drop_counties:
        GH_COUNTIES_DATA.drop(GH_COUNTIES_DATA[GH_COUNTIES_DATA['state'] == state].index, inplace=True)
    compute_counties_data()
    GH_COUNTIES_DATA.to_csv('us-counties.csv')
else:
    GH_COUNTIES_DATA = pd.read_csv('us-counties.csv', parse_dates=['date', 'avg_dates'])

STATES = sorted(GH_STATES_DATA['state'].unique())
COUNTIES = sorted({f'{state}, {county}' for county, state in zip(GH_COUNTIES_DATA['county'], GH_COUNTIES_DATA['state'])})


# In[8]:


TRACKING_DATA = pd.DataFrame.from_dict(requests.get(url='https://covidtracking.com/api/v1/states/daily.json').json())

TRACKING_DATA['datetime'] = [datetime.strptime(str(x), '%Y%m%d') for x in TRACKING_DATA['date']]
TRACKING_DATA['positivity'] = TRACKING_DATA['positive'] / TRACKING_DATA['totalTestResults'] * 100

STATE_ABBRV = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
               'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
               'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
               'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
               'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO',
               'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
               'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH',
               'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
               'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
               'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}


# In[9]:


def compute_log_palette(palette, low, high, value):
    if np.isnan(value):
        return 'gray'
    if value >= high:
        return palette[-1]
    if value < low:
        return palette[0]
    diff = np.log(value) - np.log(low)
    key = int(diff * len(palette) / (np.log(high) - np.log(low)))
    return palette[key]


# In[10]:


def compute_linear_palette(palette, low, high, value):
    if np.isnan(value):
        return 'gray'
    if value >= high:
        return palette[-1]
    if value < low:
        return palette[0]
    diff = value - low
    key = int(diff * len(palette) / (high - low))
    return palette[key]


# In[11]:


def get_dataset(region):

    pop_entry = get_pop_entry(region)
    
    if pop_entry['GEO_ID'].values[0].startswith('04'):
        data = GH_STATES_DATA[GH_STATES_DATA['state'] == region]
    elif pop_entry['GEO_ID'].values[0].startswith('05'):
        state, county = region.split(', ')
        data = GH_COUNTIES_DATA[(GH_COUNTIES_DATA['state'] == state).values & (GH_COUNTIES_DATA['county'] == county).values]

    return data


# In[12]:


def get_data(region, per_capita=False, data_type='cases', constant_date=None):

    data = dict()
    test_data = None

    if data_type in ('cases', 'deaths'):

        subset = get_dataset(region)
        
        dates = subset['date']
        avg_dates = subset['avg_dates']
        
        if not per_capita:
            dt_label = data_type
            label = f'Total New {data_type.title()}'
        else:
            dt_label = f'{data_type}_pc'
            label = f'New {data_type.title()} per 100,000'

        data = subset[f'diff_{dt_label}']
        avg_data = subset[f'avg_{dt_label}']

    elif data_type in ('positivity', 'constant positivity', 'constant testing'):

        subset = TRACKING_DATA[TRACKING_DATA['state'] == STATE_ABBRV[region]].sort_values('date')
        
        date_offset = np.timedelta64(3, 'D') + np.timedelta64(12, 'h')
        
        dates = subset['datetime']
        avg_dates = dates - date_offset
        
        if data_type == 'positivity':
            data = subset['positivity']
            label = 'Positivity (%)'
        elif data_type == 'constant positivity':
            positivity = subset[subset['datetime'] == constant_date]['positivity'].values
            data = subset['positiveIncrease']
            test_data = (subset['totalTestResults'] * positivity / 100).diff().rolling(7).mean()
            label = 'Cases'
        elif data_type == 'constant testing':
            total_tests = subset[subset['datetime'] == constant_date]['totalTestResultsIncrease'].values
            data = subset['positiveIncrease']
            test_data = (subset['positivity'] * total_tests / 100).rolling(7).mean()
            label = 'Cases'

        if data_type != 'positivity' and per_capita:
            pop = population(region)
            data = data / pop * 100000
            
        avg_data = data.rolling(7).mean()

        if data_type != 'positivity':
            if per_capita:
                label = f'New {label.title()} per 100,000'
            else:
                label = f'Total New {label.title()}'

    return dates, avg_dates, data, avg_data, test_data, label


# In[13]:


class StateDisplay:
    
    def __init__(self, dataset=STATES):

        self.dataset = dataset

        self.state_selection = MultiSelect(title='States:', options=self.dataset, value=['New York', 'Texas'], height=550)
        self.per_capita = RadioGroup(labels=['Total', 'Per Capita'], active=0)
        self.data_getter = RadioGroup(labels=['Cases', 'Deaths', 'Positivity', 'Constant Positivity',
                                              'Constant Testing'], active=0)
        self.plot_type = RadioGroup(labels=['Linear', 'Logarithmic'], active=0)
        
        self.constant_date = DatePicker(title='Constant Date', value=(datetime.today() - timedelta(days=1)).date())
        
        self.src = None
        self.p = None
        self.logp = None
    
    def make_dataset(self, state_list):

        by_state = {}

        color_cycle = cycle(Category20_20)
        palette = [next(color_cycle) for _ in self.dataset]

        for state_name in state_list:

            per_capita = self.per_capita.active == 1
            data_getter = self.data_getter.labels[self.data_getter.active].lower()
            constant_date = self.constant_date.value

            dates, avg_dates, data, avg_data, test_data, label = get_data(state_name, per_capita, data_getter, constant_date)

            by_state.setdefault('avg_date', []).append(avg_dates.values)
            by_state.setdefault('avg_data', []).append(avg_data.values)

            by_state.setdefault('state', []).append(state_name)
            by_state.setdefault('color', []).append(palette[self.dataset.index(state_name)])

        return label, ColumnDataSource(by_state)
    
    def make_plot(self):

        self.p = figure(title='COVID-19 Cases', x_axis_label='Date',
                        x_axis_type='datetime', y_axis_label='Total Cases')
            
        self.p.multi_line(source=self.src, xs='avg_date', ys='avg_data',
                          legend_field='state', color='color', line_width=2)

        self.p.legend.location = 'top_left'
    
        self.logp = figure(title='COVID-19 Cases', x_axis_label='Date',
                           x_axis_type='datetime', y_axis_label='Total Cases',
                           y_axis_type = 'log')
            
        self.logp.multi_line(source=self.src, xs='avg_date', ys='avg_data',
                             legend_field='state', color='color', line_width=2)

        self.logp.legend.location = 'top_left'

    def update(self, attr, old, new):

        states_to_plot = sorted(self.state_selection.value)

        label, new_src = self.make_dataset(states_to_plot)

        if self.src is None:
            self.src = new_src
            self.make_plot()
        else:
            self.src.data.update(new_src.data)

        if self.plot_type.active == 0:
            self.p.visible = True
            self.logp.visible = False
        else:
            self.p.visible = False
            self.logp.visible = True

        self.p.yaxis.axis_label = label
        self.logp.yaxis.axis_label = label
                
    def run(self):

        self.state_selection.on_change('value', self.update)
    
        self.per_capita.on_change('active', self.update)
        self.data_getter.on_change('active', self.update)
        self.plot_type.on_change('active', self.update)
        self.constant_date.on_change('value', self.update)

        controls = column([self.state_selection, self.per_capita, self.data_getter, self.plot_type,
                           self.constant_date])

        self.update(None, None, None)
        
        plots = column(self.p, self.logp)

        return row(controls, plots)


# In[14]:


# show(Application(FunctionHandler(StateDisplay().run)))

# In[15]:


class SingleStateDisplay:
    
    def __init__(self):
        
        self.state = 'New York'
        self.menu = STATES

        self.state_selection = Dropdown(menu=self.menu, label=self.state)
        self.per_capita = RadioGroup(labels=['Total', 'Per Capita'], active=0)
        self.data_getter = RadioGroup(labels=['Cases', 'Deaths', 'Positivity', 'Constant Positivity',
                                              'Constant Testing'], active=0)
        self.plot_type = RadioGroup(labels=['Linear', 'Logarithmic'], active=0)
        
        self.constant_date = DatePicker(title='Constant Date', value=(datetime.today() - timedelta(days=1)).date())
        self.show_constant_date = True

        self.src = None
        self.p = None
        self.logp = None
    
    def make_dataset(self, state_name=''):

        per_capita = self.per_capita.active == 1
        data_getter = self.data_getter.labels[self.data_getter.active].lower()
        constant_date = self.constant_date.value

        dates, avg_dates, data, avg_data, test_data, label = get_data(state_name, per_capita, data_getter, constant_date)

        data_dict = {'date': dates.values, 'avg_date': avg_dates.values, 'data': data.values, 'avg_data': avg_data.values}
        
        if test_data is None:
            data_dict['test_data'] = data.values.copy()
            data_dict['test_data'][:] = np.NaN
        else:
            data_dict['test_data'] = test_data.values

        return label, ColumnDataSource(data_dict)
    
    def make_plot(self):

        self.p = figure(title='COVID-19 Cases', x_axis_label='Date',
                        x_axis_type='datetime', y_axis_label='Total Cases')
            
        self.p.vbar(source=self.src, x='date', top='data', color='orange')
        self.p.line(source=self.src, x='avg_date', y='avg_data', line_width=2)
        self.p.line(source=self.src, x='date', y='test_data', line_width=2, line_dash='dashed')
        
        self.p.legend.visible = False

        self.logp = figure(title='COVID-19 Cases', x_axis_label='Date',
                           x_axis_type='datetime', y_axis_label='Total Cases',
                           y_axis_type='log')
            
        self.logp.vbar(source=self.src, x='date', bottom=1e-10, top='data', color='orange')
        self.logp.line(source=self.src, x='avg_date', y='avg_data', line_width=2)
        self.logp.line(source=self.src, x='date', y='test_data', line_width=2, line_dash='dashed')
        
        self.logp.legend.visible = False
    
    def update(self, attr, old, new):

        label, new_src = self.make_dataset(self.state)

        if self.src is None:
            self.src = new_src
            self.make_plot()
        else:
            self.src.data.update(new_src.data)

        if self.plot_type.active == 0:
            self.p.visible = True
            self.logp.visible = False
        else:
            self.p.visible = False
            self.logp.visible = True

        self.p.yaxis.axis_label = label
        self.logp.yaxis.axis_label = label
                
    def update_selection(self, event):
        self.state = event.item
        self.state_selection.label = self.state
        self.update(None, None, None)

    def run(self):

        self.state_selection.on_click(self.update_selection)
    
        self.per_capita.on_change('active', self.update)
        self.data_getter.on_change('active', self.update)
        self.plot_type.on_change('active', self.update)
        self.constant_date.on_change('value', self.update)

        controls = [self.state_selection, self.per_capita, self.data_getter, self.plot_type]
        if self.show_constant_date:
            controls.append(self.constant_date)

        controls = column(controls)

        self.update_selection(MenuItemClick(None, self.state))

        plots = column(self.p, self.logp)

        return row(controls, plots)


# In[16]:


# show(Application(FunctionHandler(SingleStateDisplay().run)))


# In[17]:


class CountyDisplay(StateDisplay):
    
    def __init__(self):

        super().__init__(COUNTIES)

        self.state_selection.title = 'Counties:'
        self.state_selection.value = ['New York, Washington', 'Texas, Harris']
        
        self.data_getter.labels = ['Cases', 'Deaths']
        
        self.show_constant_date = False


# In[18]:


# show(Application(FunctionHandler(CountyDisplay().run)))


# In[19]:


class MapBase:
    
    def __init__(self):
        
        self.per_capita = RadioGroup(labels=['Total', 'Per Capita', 'Logarithmic'], active=0, width=100)
        self.data_getter = RadioGroup(labels=['Cases', 'Deaths', 'Positivity'], active=0, width=100)
        self.date = DatePicker(title='Date', width=200)
        self.save_files = CheckboxGroup(labels=['Save files'])
    
        self.tooltips = [('Name', '@name'),
                         ('Value', '@value')]

        self.src = None
        self.p = None

        self.doc = None
        self.button = None
        self.callback = None
        self.counter = None
        
        self.tempdir = None
        self.filenames = None

    def make_dataset(self):
        raise NotImplementedError

    def make_plot(self, maxval):
        
        color_mapper = LinearColorMapper(palette=PALETTE, low=0, high=maxval)

        color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(),
                             label_standoff=12, border_line_color=None, location=(0,0))
        
        self.p = figure(toolbar_location="left", plot_width=950, aspect_ratio=1.8, tooltips=self.tooltips)

        self.p.patches(source=self.src, xs='lons', ys='lats', fill_color='color',
                       line_color='white', line_width=0.5)
        
        self.p.axis.visible = False
        self.p.grid.visible = False
        self.p.outline_line_color = None

        self.p.add_layout(color_bar, 'right')

    def update(self, attr, old, new):

        label, maxval, new_src = self.make_dataset()

        if self.src is None:
            self.src = new_src
            self.make_plot(maxval)
        else:
            self.src.data.update(new_src.data)

        self.p.title.text = f'{label} on {date.fromisoformat(self.date.value).strftime("%B %d, %Y")}'
        
        color_mapper = LogColorMapper  # if self.per_capita.active == 2 else LinearColorMapper
            
        self.p.right[0].color_mapper = color_mapper(palette=PALETTE, low=0, high=maxval)
        self.p.right[0].ticker = BasicTicker()
        
    def animate_update(self):

        self.counter += 1
        
        if self.save_files.active == [0]:
            filename = os.path.join(self.tempdir, f'{self.__class__.__name__}_plot_{self.counter}.png')
            export_png(self.p, filename=filename)
            self.filenames.append(filename)
        
        new_date = date.fromisoformat(self.date.value) + timedelta(days=1)
        
        self.date.value = new_date.isoformat()
        
        if new_date > self.date.enabled_dates[0][1] - timedelta(days=1):
            self.animate()
    
    def animate(self):
        
        if self.button.label == '► Play':
        
            self.button.label = '❚❚ Pause'
            
            self.counter = 0
            
            if self.save_files.active == [0]:
                self.tempdir = tempfile.mkdtemp()
                self.filenames = []
            
            self.callback = curdoc().add_periodic_callback(self.animate_update, 200)
        
        else:
            
            self.button.label = '► Play'
            
            curdoc().remove_periodic_callback(self.callback)
            
            if self.save_files.active == [0]:
                with imageio.get_writer(f'{self.__class__.__name__}_plot.gif', mode='I') as writer:
                    for filename in self.filenames:
                        image = imageio.imread(filename)
                        writer.append_data(image)
                shutil.rmtree(self.tempdir)

    def run(self):
        
        self.per_capita.on_change('active', self.update)
        self.data_getter.on_change('active', self.update)
        self.date.on_change('value', self.update)

        self.update(None, None, None)

        self.button = Button(label='► Play', width=60)
        self.button.on_click(self.animate)

        controls = row([self.per_capita, self.data_getter, self.date, self.save_files, self.button])
        return column(self.p, controls)


# In[20]:


class StateMap(MapBase):
    
    def __init__(self):

        super().__init__()
    
        self.tooltips = [('State', '@state'),
                         ('Value', '@value')]

        dates = GH_STATES_DATA.loc[:, 'date']
        self.date.value = dates.max().date()
        self.date.enabled_dates = [(dates.min().date(), dates.max().date())]

    def make_dataset(self):

        per_capita = self.per_capita.active == 1
        logarithmic = self.per_capita.active == 2
        data_type = self.data_getter.labels[self.data_getter.active].lower()
        date = self.date.value

        data = np.empty(len(US_STATES))

        if data_type in ('cases', 'deaths'):

            if not per_capita:
                dt_label = data_type
                label = f'Total New {data_type.title()}'
            else:
                dt_label = f'{data_type}_pc'
                label = f'New {data_type.title()} per 100,000'

            subset = GH_STATES_DATA.loc[GH_STATES_DATA['date'] == date, :]
            for i, (abbrv, state) in enumerate(US_STATES.items()):
                state_name = state['name']
                value = subset.loc[subset['state'] == state_name, f'avg_{dt_label}']
                if not value.empty and not np.isnan(value.values[0]):
                    data[i] = max(0, value.values[0])
                else:
                    data[i] = 0

            maxval = GH_STATES_DATA.loc[:, f'avg_{dt_label}'].max()

        elif data_type == 'positivity':

            label = 'Positivity (%)'

            subset = TRACKING_DATA.loc[TRACKING_DATA['datetime'] == date, ('state', 'positivity')]
            for i, (abbrv, state) in enumerate(US_STATES.items()):
                value = subset.loc[subset['state'] == abbrv.upper(), 'positivity']
                if not value.empty and not np.isnan(value.values[0]):
                    data[i] = max(0, value.values[0])
                else:
                    data[i] = 0

            maxval = TRACKING_DATA.loc[:, 'positivity'].max()

        interp = compute_log_palette  # if logarithmic else compute_linear_palette

        color_data = {'color': [interp(PALETTE, maxval / 256, maxval, val) for val in data],
                      'value': data,
                      'state': [state['name'] for state in US_STATES.values()]}

        for state in US_STATES.values():
            color_data.setdefault('lons', []).append(state['lons'])
            color_data.setdefault('lats', []).append(state['lats'])
            
        return label, maxval, ColumnDataSource(color_data)


# In[21]:


# show(Application(FunctionHandler(StateMap().run)))


# In[35]:


class CountyMap(MapBase):
    
    def __init__(self):
        
        super().__init__()

        dates = GH_COUNTIES_DATA.loc[:, 'date']
        self.date.value = dates.max().date()
        self.date.enabled_dates = [(dates.min().date(), dates.max().date())]
        
        self.data_getter.labels = ['Cases', 'Deaths']
    
        self.tooltips = [('Name', '@name'),
                         ('Cases', '@cases'),
                         ('Deaths', '@deaths'),
                         ('Cases per Cap', '@cases_pc'),
                         ('Deaths per Cap', '@deaths_pc'),
                         ('Pop', '@population')]
    
    def make_dataset(self):

        per_capita = self.per_capita.active == 1
        data_type = self.data_getter.labels[self.data_getter.active].lower()
        date = self.date.value

        excluded = ('ak', 'hi', 'pr', 'gu', 'vi', 'mp', 'as')
        counties = {abbrv: county for abbrv, county in US_COUNTIES.items()
                    if county['state'] not in excluded}

        data = np.zeros(len(counties), dtype=float)
        cases = np.zeros(len(counties), dtype=float)
        deaths = np.zeros(len(counties), dtype=float)
        cases_pc = np.zeros(len(counties), dtype=float)
        deaths_pc = np.zeros(len(counties), dtype=float)
        pop = np.zeros(len(counties), dtype=int)

        if not per_capita:
            dt_label = data_type
            label = f'Total New {data_type.title()}'
        else:
            dt_label = f'{data_type}_pc'
            label = f'New {data_type.title()} per 100,000'

        subset = GH_COUNTIES_DATA.loc[GH_COUNTIES_DATA['date'] == date, :]
        for i, (abbrv, county) in enumerate(counties.items()):
            state_name, county_name = parse_detailed_name(county['detailed name'])
            value = subset.loc[(subset['county'] == county_name).values &
                               (subset['state'] == state_name).values, :]
            if not value.empty:
                dataval = value[f'avg_{dt_label}'].values[0]
                if not np.isnan(dataval):
                    data[i] = max(0, dataval)
                else:
                    data[i] = 0
                cases[i] = value[f'avg_cases'].values[0]
                deaths[i] = value[f'avg_deaths'].values[0]
                cases_pc[i] = value[f'avg_cases_pc'].values[0]
                deaths_pc[i] = value[f'avg_deaths_pc'].values[0]
                pop[i] = population(f'{state_name}, {county_name}')

        if per_capita and data_type != 'deaths':
            maxval = 1000
        else:
            maxval = GH_COUNTIES_DATA.loc[:, f'avg_{dt_label}'].max()

        interp = compute_log_palette  # if logarithmic else compute_linear_palette

        color_data = {'color': [interp(PALETTE, maxval / 256, maxval, val) for val in data],
                      'value': data,
                      'cases': cases,
                      'deaths': deaths,
                      'cases_pc': cases_pc,
                      'deaths_pc': deaths_pc,
                      'population': pop,
                      'name': [county['detailed name'] for county in counties.values()]}

        for county in counties.values():
            color_data.setdefault('lons', []).append(county['lons'])
            color_data.setdefault('lats', []).append(county['lats'])
            
        return label, maxval, ColumnDataSource(color_data)
    
    def make_plot(self, maxval):
        
        super().make_plot(maxval)
        
        state_xs = [state['lons'] for state in US_STATES.values()]
        state_ys = [state['lats'] for state in US_STATES.values()]

        #self.p.patches(state_xs, state_ys, fill_alpha=0.0, line_color="#884444", line_width=2, line_alpha=0.3)


# In[36]:


# show(Application(FunctionHandler(CountyMap().run)))


# In[ ]:


tab1 = Panel(child=StateDisplay().run(), title='State Comparisons')
tab2 = Panel(child=CountyDisplay().run(), title='County Comparisons')
tab3 = Panel(child=SingleStateDisplay().run(), title='State Data')
tab4 = Panel(child=StateMap().run(), title='State Map')
tab5 = Panel(child=CountyMap().run(), title='County Map')

tabs = Tabs(tabs=[tab1, tab2, tab3, tab4, tab5])

curdoc().add_root(tabs)